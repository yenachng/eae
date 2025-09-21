# tests/test_data.py
import numpy as np
from src.preprocessing.data import zscore_per_channel, MultiTaskWindows
from src.data.helpers import subjects_passive, load_raws, bids_path_for
from src.data import config
from src.preprocessing.clean_raws import BuildArrays

# subs_all_passives = subjects_passive("all")
bids_root = config.BIDS_ROOT

subject_0 = "NDARAC904DMU"

# group eeg recordings for subject 0 by task

paths_by_task = {
    "rest": bids_path_for(bids_root=bids_root, subjects=[subject_0], tasks=["RestingState"]),
    "passive": bids_path_for(bids_root=bids_root, subjects=[subject_0], tasks=config.PASSIVE_TASKS),
    "active": bids_path_for(bids_root=bids_root, subjects=[subject_0], tasks=config.ACTIVE_TASKS)
}
print("paths by task dict:")
for tk, paths in paths_by_task.items():
    print(f"{tk:}\n {paths}")
builder = BuildArrays()
builder.load_arrays_by_task(subject_0, paths_by_task)