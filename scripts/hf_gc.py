"""Free disk from the Hugging Face cache, including the shared hub/blobs store.

    python scripts/hf_gc.py                      # dry run: list weight files nothing points to
    python scripts/hf_gc.py --delete             # delete them
    python scripts/hf_gc.py Qwen/Qwen3-8B --delete   # remove that model first, then the weights nothing points to

A model's folder (hub/models--Org--Name) only holds symlinks; the weights live in blobs/. Deleting the folder frees nothing
until the blobs it pointed to are gone. Files modified in the last 10 minutes and *.incomplete downloads are never touched.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import time
from pathlib import Path


def hub_dir() -> Path:
    for v in ("HF_HUB_CACHE",):
        if os.environ.get(v):
            return Path(os.environ[v])
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="*", help="model ids to remove first, e.g. Qwen/Qwen3-8B")
    ap.add_argument("--delete", action="store_true")
    a = ap.parse_args(argv)
    hub = hub_dir()
    for r in a.repos:
        d = hub / ("models--" + r.replace("/", "--"))
        if d.exists():
            print(("removing " if a.delete else "would remove ") + str(d))
            if a.delete:
                shutil.rmtree(d)
    used = set()
    for link in glob.glob(str(hub / "models--*" / "snapshots" / "**"), recursive=True):
        if os.path.islink(link):
            used.add(os.path.realpath(link))
    blob_dirs = [hub / "blobs", *map(Path, glob.glob(str(hub / "models--*" / "blobs")))]
    freed = 0
    for bd in blob_dirs:
        for root, _dirs, files in os.walk(bd):            # blobs may sit in shard folders (blobs/7f/<hash>)
            for fn in files:
                b = os.path.join(root, fn)
                if b.endswith(".incomplete") or os.path.realpath(b) in used:
                    continue
                if time.time() - os.path.getmtime(b) < 600:
                    continue
                size = os.path.getsize(b)
                freed += size
                print(f"{'delete' if a.delete else 'orphan'} {b} ({size / 1e9:.2f} GB)")
                if a.delete:
                    os.remove(b)
    print(f"{'freed' if a.delete else 'would free'} {freed / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
