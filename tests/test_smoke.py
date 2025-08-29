import importlib

def test_imports():
    assert importlib.import_module("clip_causal_repair") is not None
    assert importlib.import_module("clip_causal_repair.clip_loader") is not None
