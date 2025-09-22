from src.preprocessing.windows.multitask_windows import MultiTaskWindows
from src.preprocessing.clean_raws import BuildArrays

subject_0 = "NDARAC904DMU"
builder = BuildArrays()

arrays_by_task = builder.arrays_by_task(subject_0)

mtw = MultiTaskWindows(arrays_by_task, window_size=)