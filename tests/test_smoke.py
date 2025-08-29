import importlib

def test_imports():
    # Base package import
    assert importlib.import_module("clip_causal_repair") is not None

    # Submodules
    assert importlib.import_module("clip_causal_repair.clip_loader") is not None
    assert importlib.import_module("clip_causal_repair.utils") is not None
    assert importlib.import_module("clip_causal_repair.trainer") is not None
    assert importlib.import_module("clip_causal_repair.model") is not None
    assert importlib.import_module("clip_causal_repair.evaluator") is not None
    assert importlib.import_module("clip_causal_repair.preprocessor") is not None
