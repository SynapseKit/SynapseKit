from .base import BaseCheckpointer
from .json_file import JSONFileCheckpointer
from .memory import InMemoryCheckpointer
from .sqlite import SQLiteCheckpointer

__all__ = [
    "BaseCheckpointer",
    "InMemoryCheckpointer",
    "JSONFileCheckpointer",
    "PostgresCheckpointer",
    "RedisCheckpointer",
    "SQLiteCheckpointer",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    _lazy = {
        "RedisCheckpointer": "redis",
        "PostgresCheckpointer": "postgres",
    }
    if name in _lazy:
        import importlib

        mod = importlib.import_module(f".{_lazy[name]}", __name__)
        cls = getattr(mod, name)
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
