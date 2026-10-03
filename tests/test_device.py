from types import SimpleNamespace

from recl2bench.rerankers.cross_encoder import pick_device, pick_dtype


def _torch(cuda, mps):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda),
                           backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
                           float16="f16", float32="f32")


def test_device_order():
    assert pick_device(_torch(True, True)) == "cuda"
    assert pick_device(_torch(False, True)) == "mps"
    assert pick_device(_torch(False, False)) == "cpu"
    assert pick_device(_torch(True, True), "cpu") == "cpu"


def test_dtype():
    t = _torch(False, True)
    assert pick_dtype(t, "mps") == "f16" and pick_dtype(t, "cpu") == "f32"
