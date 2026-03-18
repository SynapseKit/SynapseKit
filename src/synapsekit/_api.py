"""API stability markers for SynapseKit public surface."""

from __future__ import annotations

import functools
import warnings
from typing import Any, TypeVar

F = TypeVar("F")


def public_api(obj: F) -> F:
    """Mark a class or function as part of the stable public API.

    This is purely a documentation/metadata marker; it does not modify
    the wrapped object's behaviour.
    """
    obj._synapsekit_public_api = True  # type: ignore[attr-defined]
    return obj


def experimental(obj: F) -> F:
    """Mark a class or function as experimental.

    A ``FutureWarning`` is emitted the first time the decorated
    callable is used, so downstream users know the API may change.
    """
    if isinstance(obj, type):
        # Wrap __init__ so warning fires on instantiation
        original_init = obj.__init__

        @functools.wraps(original_init)
        def warned_init(self: Any, *args: Any, **kwargs: Any) -> None:
            warnings.warn(
                f"{obj.__qualname__} is experimental and may change without notice.",
                FutureWarning,
                stacklevel=2,
            )
            original_init(self, *args, **kwargs)

        obj.__init__ = warned_init  # type: ignore[attr-defined]
        obj._synapsekit_experimental = True  # type: ignore[attr-defined]
        return obj  # type: ignore[return-value]

    @functools.wraps(obj)  # type: ignore[arg-type]
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"{obj.__qualname__} is experimental and may change without notice.",  # type: ignore[union-attr]
            FutureWarning,
            stacklevel=2,
        )
        return obj(*args, **kwargs)  # type: ignore[misc]

    wrapper._synapsekit_experimental = True  # type: ignore[attr-defined]
    return wrapper  # type: ignore[return-value]


def deprecated(reason: str, alternative: str | None = None):
    """Mark a class or function as deprecated.

    Parameters
    ----------
    reason:
        Why the API is deprecated.
    alternative:
        Suggested replacement (included in the warning message).
    """
    msg = f"{{name}} is deprecated: {reason}"
    if alternative:
        msg += f" Use {alternative} instead."

    def decorator(obj: F) -> F:
        if isinstance(obj, type):
            original_init = obj.__init__

            @functools.wraps(original_init)
            def warned_init(self: Any, *args: Any, **kwargs: Any) -> None:
                warnings.warn(
                    msg.format(name=obj.__qualname__),
                    DeprecationWarning,
                    stacklevel=2,
                )
                original_init(self, *args, **kwargs)

            obj.__init__ = warned_init  # type: ignore[attr-defined]
            obj._synapsekit_deprecated = True  # type: ignore[attr-defined]
            return obj  # type: ignore[return-value]

        @functools.wraps(obj)  # type: ignore[arg-type]
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                msg.format(name=obj.__qualname__),  # type: ignore[union-attr]
                DeprecationWarning,
                stacklevel=2,
            )
            return obj(*args, **kwargs)  # type: ignore[misc]

        wrapper._synapsekit_deprecated = True  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator
