import mne
import numpy as np
import torch

def build_epochs_all_channels(raw: mne.io.BaseRaw, L: int = 512, S: int = 512) -> mne.Epochs:
    # builds fixed-length epochs with window length L and stride S
    if L <= 0 or S <= 0:
        raise ValueError("L and S must be positive.")
    sfreq = float(raw.info["sfreq"])
    events = mne.make_fixed_length_events(
        raw, id=1, start=0.0, stop=None, duration=S / sfreq, first_samp=False
    )
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0.0,
        tmax=(L - 1) / sfreq,
        picks="eeg",
        baseline=None,
        preload=False,
        reject_by_annotation=True,
        verbose=False,
    )
    return epochs

class EpochsDataset(torch.utils.data.Dataset):
    # yields (n_ch, L) in mV
    def __init__(self, epochs: mne.Epochs, dtype: torch.dtype = torch.float32):
        self.epochs = epochs
        self.dtype = dtype
        self.starts = self.epochs.events[:, 0].astype(np.int64)
        # sanity check window length
        x0 = self.epochs[0].get_data(units="uV")[0]
        if x0.shape[-1] <= 0:
            raise RuntimeError("empty window length")
        self._L = x0.shape[-1]

    def __len__(self) -> int:
        return len(self.epochs)

    def __getitem__(self, i: int):
        x = self.epochs[i].get_data(units="uV")[0]  # (n_ch, L)
        if x.shape[-1] != self._L:
            raise RuntimeError("inconsistent window length")
        t = torch.as_tensor(x, dtype=self.dtype).contiguous()
        start = int(self.starts[i])
        return {"x": t, "start": start, "end": start + t.shape[-1]}  # end is exclusive
