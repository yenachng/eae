# quick demo: list subjects and load raws for passive tasks

from __future__ import annotations
from .data.helpers import subjects_passive, bids_path_passive, load_raws

def main() -> None:
    subs = subjects_passive(require="all")
    print(f"{len(subs)} subjects with all passive tasks")
    bps = bids_path_passive(require="all")
    print(f"{len(bps)} eeg files resolved")
    for i, (bp, raw) in enumerate(load_raws(bps, preload=False), 1):
        # minimal probe to confirm load without memory blowup
        print(f"[{i:04d}] sub-{bp.subject} | task-{bp.task} | nchan={raw.info['nchan']} | sfreq={raw.info['sfreq']}")
        raw.close()  # keep memory low

if __name__ == "__main__":
    main()
