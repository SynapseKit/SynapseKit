"""Every public export must actually resolve.

`synapsekit.__all__` is the promised top-level API. A name listed there but
missing from the lazy-import map (`_LAZY_IMPORTS`) raises `AttributeError` on
access, which breaks `from synapsekit import X` and `from synapsekit import *`
even though the underlying class exists in its subpackage. `SupabaseLoader` was
exactly this: real class in `synapsekit.loaders`, listed in `__all__`, but never
wired at the top level.

A *missing optional dependency* is different — it surfaces as
`ModuleNotFoundError` (the extra just isn't installed here) and is tolerated. An
`AttributeError` is always a wiring bug.
"""

from __future__ import annotations

import synapsekit


def test_all_exports_resolve() -> None:
    broken = []
    for name in synapsekit.__all__:
        try:
            getattr(synapsekit, name)
        except ModuleNotFoundError:
            # Optional extra not installed in this environment — not a wiring bug.
            pass
        except AttributeError as exc:
            broken.append(f"{name}: {exc}")

    assert not broken, "names in synapsekit.__all__ that do not resolve:\n" + "\n".join(broken)
