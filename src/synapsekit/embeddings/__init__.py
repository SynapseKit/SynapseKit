from .backend import SynapsekitEmbeddings

__all__ = ["ONNXEmbeddings", "SynapsekitEmbeddings"]

_BACKENDS = {
    "ONNXEmbeddings": ".onnx",
}


def __getattr__(name: str):
    if name in _BACKENDS:
        import importlib

        mod = importlib.import_module(_BACKENDS[name], __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
