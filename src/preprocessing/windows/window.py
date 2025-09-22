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


class MultiTaskWindows(torch.utils.data.Dataset):
    # pools windows across tasks
    '''
    pools overlapping windows across tasks
    input: {task: array (n_rec, n_ch, n_time}
    returns: (float tensor (c,t), long task_id)
    '''
    def __init__(self, arrays_by_task: dict[str, np.ndarray], window_size:int, stride: int):
        # arrays_by_task {task_name : array of shape (n_rec, n_ch, n_time)}
        assert window_size > 0 and stride > 0
        
        self.ws = int(window_size)
        self.st = int(stride)
        self.meta = []

        self.task_names = sorted(arrays_by_task.keys())
        self.task_to_id = {k:i for i, k in enumerate(self.task_names)}

        self.X = {k: zscore_per_channel(np.ascontiguousarray(v, dtype=np.float32, copy=False), axis=-1) for k,v in arrays_by_task.items()}

        # build flat index
        meta = []
        for k, arr in self.X.items():
            tid = self.task_to_id[k]
            assert arr.ndim==3, "expected (n_rec, n_ch, n_time)"
            n_rec, _, n_time = arr.shape
            
            starts = np.arange(0, n_time-self.ws+1, self.st, dtype=np.int64)
            for r in range(n_rec):
                n_time = arr[r].shape[-1]
                if n_time < self.ws:
                    print(f"[{k} (tid:{tid})] recording shorter than ws; skipping")
                    continue
                for s in starts:
                    meta.append((tid, r, int(s), int(s)+self.ws))
        self.meta = np.asarray(meta, dtype=np.int64)


    def __len__(self) -> int:
        # how many samples total
        return int(self.meta.shape[0])


    def __getitem__(self, i:int):
        # return (FloatTensor(c,t), LongTensor())
        tid, r, s, t = self.meta[i]
        arr = self.X[self.task_names[tid]]
        x = arr[r, :, s : t]
        x_t = torch.from_numpy(x).contiguous()
        tid_t = torch.tensor(tid, dtype=torch.long)
        return {"x": x_t, "task_id":tid_t, "rec_idx":torch.tensor(r, dtype=torch.long), "start":torch.tensor(s, dtype=torch.long), "end":torch.tensor(t, dtype=torch.long)}

    def __repr__(self):
        summary = f"summary:\nwindow count: {self.__len__()}\nwindow size: {self.ws}\nstride: {self.st}\ntask name count: {len(self.task_names)}\ntask names: {self.task_names}"
        return summary
    