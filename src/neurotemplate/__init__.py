"""NeuroTemplate: system identification and control from a plant's time-series record."""
import os

# neuromancer imports pyts, which compiles numba kernels as it is imported
# (20-60 s per process), for one neuromancer plot this library never calls.
# Set before anything imports numba. NUMBA_DISABLE_JIT=0 in the environment
# keeps the JIT for your own numba code in the same process.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

# `torch.set_num_threads(1)`, which is what `threads: 1` calls, leaves OpenMP's
# workers spinning: in a 2-CPU container that cost 1.7x to 5.6x. Only this
# variable, read when torch loads, prevents them. A run's own `threads` still
# applies on top of it. Import this package before torch for it to take effect.
os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "0.2.0"
