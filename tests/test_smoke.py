import importlib

def test_imports():
    # Base package import
    assert importlib.import_module("clip_causal_repair") is not None

