from typing import Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler

def temporal_split(
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
) -> Tuple[tuple, tuple, tuple]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

    n = len(X)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def fit_scaler_on_train(X_train: np.ndarray) -> StandardScaler:
    n_samples, window, n_features = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))
    return scaler

def transform_windows(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n_samples, window, n_features = X.shape
    X_scaled = scaler.transform(X.reshape(-1, n_features))
    return X_scaled.reshape(n_samples, window, n_features)