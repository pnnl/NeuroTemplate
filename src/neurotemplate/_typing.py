"""Runtime type-checking toggle shared across the package.

`beartype` decorates functions with runtime type checks, but only when the
`RUNTIME_TYPING` environment variable is set. Otherwise it is a no-op
passthrough so production runs pay no per-call overhead. Import it as:

    from neurotemplate._typing import beartype
"""
from __future__ import annotations

import os

if os.environ.get("RUNTIME_TYPING", 0):
    from beartype import beartype as beartype
else:
    def beartype(fn):
        return fn
