"""Shared pytest setup.

The library is installed as `neurotemplate` (see pyproject.toml [build-system]);
test imports go through `from neurotemplate.* import ...`.

A `fixtures/` directory holds stub hook modules used by tests that exercise
the hook-discovery path. We prepend it to sys.path here so those stubs are
importable as if they lived in an application's working directory.
"""

import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
if FIXTURES.is_dir() and str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))
