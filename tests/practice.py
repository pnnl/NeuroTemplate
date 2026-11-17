"""The guides' practice record: `guides/make_data.py`, loaded from the template `nt init`
writes, so the tests run on the same building every guide's examples do."""
import importlib.util

from neurotemplate.init import TEMPLATE

_spec = importlib.util.spec_from_file_location("make_data", TEMPLATE / "guides" / "make_data.py")
make_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_data)
