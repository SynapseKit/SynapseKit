from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def requires_human_confirmation(reason: str | None = None) -> Callable[[F], F]:
    """Mark an action helper as requiring explicit human confirmation."""

    def decorator(func: F) -> F:
        func.__synapsekit_requires_confirmation__ = True  # type: ignore[attr-defined]
        func.__synapsekit_confirmation_reason__ = reason  # type: ignore[attr-defined]

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        wrapper.__synapsekit_requires_confirmation__ = True  # type: ignore[attr-defined]
        wrapper.__synapsekit_confirmation_reason__ = reason  # type: ignore[attr-defined]
        return cast(F, wrapper)

    return decorator


def requires_confirmation(obj: Any) -> bool:
    """Return whether an object was marked with ``requires_human_confirmation``."""

    return bool(getattr(obj, "__synapsekit_requires_confirmation__", False))
