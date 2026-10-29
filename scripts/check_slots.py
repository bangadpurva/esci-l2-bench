"""Refuse to ship a manuscript or response that still has unfilled revision slots.

    python scripts/check_slots.py paper_generated manuscript/main.tex response/*.docx

Flags, with exit code 1 if any is found:
  - \\TBD{...} calls in .tex (macro definitions and comments are ignored)
  - \\slot{name}{...} whose generated/name.tex is missing next to the .tex file
  - "[FILL" in .tex / .docx / .md
  - "n/a" cells in any tabular (a table that could not be completed)
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

NA = re.compile(r"(?<![A-Za-z])n/a(?![A-Za-z])")


def text_of(p: Path) -> str:
    if p.suffix == ".docx":
        with zipfile.ZipFile(p) as z:
            return re.sub(r"<[^>]+>", "", z.read("word/document.xml").decode("utf8", "replace"))
    return p.read_text(errors="replace")


def strip_tex(t: str) -> str:
    """Drop comments and the \\newcommand lines that define the slot macros."""
    out = []
    for line in t.splitlines():
        if line.lstrip().startswith("%") or "\\newcommand{\\TBD}" in line or "\\newcommand{\\slot}" in line:
            continue
        out.append(re.sub(r"(?<!\\)%.*$", "", line))
    return "\n".join(out)


def scan(p: Path) -> list[str]:
    out: list[str] = []
    if p.is_dir():
        for q in sorted(p.rglob("*")):
            if q.suffix in (".tex", ".docx", ".md"):
                out += scan(q)
        return out
    raw = text_of(p)
    t = strip_tex(raw) if p.suffix == ".tex" else raw
    ctx = lambda m: t[max(0, m.start() - 30):m.end() + 70].replace("\n", " ").strip()  # noqa: E731
    if p.suffix == ".tex":
        for m in re.finditer(r"\\TBD\{", t):
            out.append(f"{p.name}: \\TBD near ...{ctx(m)!r}")
        for m in re.finditer(r"\\slot\{([^}]*)\}", t):
            f = p.parent / "generated" / f"{m.group(1)}.tex"
            if not f.exists():
                out.append(f"{p.name}: table slot '{m.group(1)}' has no generated/{m.group(1)}.tex")
        if "\\begin{tabular}" in t and NA.search(t):
            out.append(f"{p.name}: table has missing cells (n/a)")
    for m in re.finditer(r"\[FILL", t):
        out.append(f"{p.name}: [FILL near ...{ctx(m)!r}")
    if p.suffix == ".tex" and p.parent.name in ("generated", "paper_generated") and NA.search(t):
        out.append(f"{p.name}: missing cells (n/a)")
    return out


if __name__ == "__main__":
    hits = [h for a in sys.argv[1:] for h in scan(Path(a))]
    print("\n".join(hits) if hits else "no unfilled slots")
    sys.exit(1 if hits else 0)
