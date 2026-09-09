"""
Deprecated: this module was renamed to `blocks.py`.

"Zone" implied HVAC where none was meant — a zone/block is a sub-model
over a group of signals, on any plant. `Zone`/`build_zones` are kept here
as aliases so code and saved specs written against the old name still
work; new code should import from `neurotemplate.modeling.blocks`.
"""

from __future__ import annotations

from neurotemplate.modeling.blocks import Block as Zone
from neurotemplate.modeling.blocks import build_blocks as build_zones

__all__ = ["Zone", "build_zones"]
