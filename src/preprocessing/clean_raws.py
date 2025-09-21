from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
import mne, mne_bids
from mne_bids import BIDSPath
import pandas as pd
import logging
import json
from src.data.helpers import load_raws

@dataclass
class CleanConfig:
    mont = mne.channels.make_standard_montage("GSN-HydroCel-128")
    ref:str = "E129"
    sfreq:float = 256.0


class BuildArrays():
    '''
    builds arrays by task
    '''
    def __init__(self):
        self.cfg = CleanConfig()

    def clean_raw(self, raw):
        cfg = self.cfg
        wanted = [f"E{i}" for i in range(1,129)]
        picks = mne.pick_channels(raw.info["ch_names"], include=wanted, ordered=True)
        raw.pick(picks)
        if "Cz" in raw.ch_names and "E129" not in raw.ch_names:
            raw.rename_channels({"Cz": "E129"})
        if "E129" in raw.ch_names:
            raw.drop_channels(["E129"])

        assert(len(raw.ch_names) == 128)

        raw.set_montage(cfg.mont)
        raw.set_eeg_reference("average")
        raw.resample(cfg.sfreq)
        
        assert(raw.info["sfreq"]==cfg.sfreq)
        X = raw.get_data().astype("float32")
        return X
        
    def load_arrays_by_task(self, subject:str, paths_by_task: Dict[str, Dict[str, List[BIDSPath]]]):
        for tk, paths_by_runs in paths_by_task.items():
            for runtype, paths in paths_by_runs.items():
                print(f"doing {runtype} tasks from {tk}")
                for bp, raw in load_raws(paths):
                    X = self.clean_raw(raw)
                    out_dir = Path("cleaned")/subject/tk
                    run = f"{int(bp.run):02d}" if runtype =="multi_run" else "01"
                    out_path = out_dir/f"{tk}_{bp.task}_{run}.npy"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(out_path, X.astype("float32"))

                    meta = {
                        "subject": bp.subject,
                        "session": bp.session or None,
                        "task": bp.task,
                        "task_type": tk,
                        "run": bp.run,
                        "sfreq": float(raw.info["sfreq"]),
                        "n_channels": X.shape[0],
                        "path_npy": str(out_path),
                        "montage": "GSN-HydroCel-128",
                        "reference": "average",
                        "dropped_ref": True
                    }
                    
                    meta_path = out_dir/f"{tk}_{bp.task}_{run}.json"
                    with open(meta_path, "w") as f:
                        json.dump(meta, f)

                    print(f"saved to {out_path},\n{meta_path}")

