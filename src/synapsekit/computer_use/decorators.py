from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def requires_human_confirmation(reason: str | None = None) -> Callable[[F], F]:
    """Mark an action helper as requiring explicit human confirmation."""

    def decorator(func: F) -> F:
        setattr(func, "__synapsekit_requires_confirmation__", True)
        setattr(func, "__synapsekit_confirmation_reason__", reason)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        setattr(wrapper, "__synapsekit_requires_confirmation__", True)
        setattr(wrapper, "__synapsekit_confirmation_reason__", reason)
        return cast(F, wrapper)

    return decorator


def requires_confirmation(obj: Any) -> bool:
    """Return whether an object was marked with ``requires_human_confirmation``."""

    return bool(getattr(obj, "__synapsekit_requires_confirmation__", False))
