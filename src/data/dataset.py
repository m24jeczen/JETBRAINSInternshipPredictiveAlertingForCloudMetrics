from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

def create_sliding_windows(
        metrics: np.ndarray,
        incident: np.ndarray,
        window_size: int,
        horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, decision_times = [],[],[]

    n = len(metrics)
    for end_idx in range(window_size, n-horizon):
        start_idx = end_idx - window_size
        future_slice = incident[end_idx:end_idx+horizon]
        label = int(future_slice.max() > 0)

        X.append(metrics[start_idx:end_idx])
        y.append(label)
        decision_times.append(end_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(decision_times)

class TimeSeriesWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]