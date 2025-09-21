import numpy as np
import torch

# eeg array of shape (n_recordings, n_channels, n_time)
def zscore_per_channel(x: np.ndarray, axis: int=-1, eps: float=1e-8)-> np.ndarray:
    '''
    returns same-shape array standardized per channel
    '''
    m = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)
    return (x-m)/(std+eps)

class multiTaskWindows(torch.utils.data.Dataset):
    # pools windows across tasks
    '''
    input : several eeg recordings grouped by task
    -> cut each recording into overlapping windows
    when asked, return one window + task label
    '''
    def __init__(self, arrays_by_task: dict[str, np.ndarray], window_size:int, stride: int):
        # arrays_by_task {task_name : array of shape (n_rec, n_ch, n_time)}
        assert window_size > 0 and stride > 0
        
        self.ws = int(window_size)
        self.st = int(stride)
        self.meta = []

        self.task_names = list(arrays_by_task.keys())
        self.task_to_id = {k:i for i, k in enumerate(self.task_names)}

        self.X = {k: zscore_per_channel(np.asarray(v, dtype=np.float32), axis=-1) for k,v in arrays_by_task.items()}

        # build flat index
        meta = []
        for k, arr in self.X.items():
            tid = self.task_to_id[k]
            assert arr.ndim==3, "expected (n_rec, n_ch, n_time)"
            n_rec, _, n_time = arr.shape
            if n_time < self.ws:
                print(f"[{k} (tid:{tid})] recording shorter than ws; skipping")
                continue
            starts = np.arange(0, n_time-self.ws+1, self.st, dtype=np.int64)
            for r in range(n_rec):
                for s in starts:
                    meta.append((tid, r, int(s)))
        self.meta = np.asarray(meta, dtype=np.int64)


    def __len__(self) -> int:
        # how many samples total
        return int(self.meta.shape[0])


    def __getitem__(self, i:int):
        # return (FloatTensor(c,t), LongTensor())
        tid, r, s = self.meta[i]
        arr = self.X[self.task_names[tid]]
        x = arr[r, :, s : s+self.ws]
        x_t = torch.from_numpy(x)
        tid_t = torch.tensor(tid, dtype=torch.long)
        return x_t, tid_t
