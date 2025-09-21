# tests/test_data.py
import numpy as np
from ..data import zscore_per_channel, multiTaskWindows

rng = np.random.default_rng(0)
C, T = 8, 4096
arrays_by_task = {
    "rest": rng.standard_normal((2, C, T)).astype("float32"),
    "active": rng.standard_normal((3, C, T)).astype("float32"),
}
arrays_by_task = {k: zscore_per_channel(v, axis=-1) for k,v in arrays_by_task.items()}
ds = multiTaskWindows(arrays_by_task, window_size=512, stride=256)
x, tid = ds[0]
assert x.ndim == 2 and x.shape[0] == C
assert tid.ndim == 0
