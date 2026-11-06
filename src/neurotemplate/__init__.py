"""NeuroTemplate: system identification and control from a plant's time-series record."""
import os

# `torch.set_num_threads(1)`, which is what `threads: 1` calls, leaves OpenMP's
# workers spinning: in a 2-CPU container that cost 1.7x to 5.6x. Only this
# variable, read when torch loads, prevents them. A run's own `threads` still
# applies on top of it. Import this package before torch for it to take effect.
os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "0.2.0"
