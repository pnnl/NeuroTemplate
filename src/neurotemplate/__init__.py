"""NeuroTemplate: system identification and control from a plant's time-series record."""
import os

# neuromancer imports pyts, which compiles numba kernels as it is imported
# (20-60 s per process), for one neuromancer plot this library never calls.
# Set before anything imports numba. NUMBA_DISABLE_JIT=0 in the environment
# keeps the JIT for your own numba code in the same process.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

__version__ = "0.2.0"
