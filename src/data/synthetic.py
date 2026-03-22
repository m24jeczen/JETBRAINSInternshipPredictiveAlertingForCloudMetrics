import numpy as np
import pandas as pd

def generate_synthetic_cloud_metrcis(
        n_steps: int,
        n_features: int,
        incident_count: int,
        incident_min_len: int,
        incident_max_len: int,
        seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    features = []
    for i in range(n_features):
        baseline = (
            0.6 * np.sin(2*np.pi*t / (120+20*i)) +
            0.3 * np.sin(2*np.pi*t / (35+5*i)) +
            0.0005*t
        )
        noise = rng.standard_t(df=3, size=n_steps) * 0.25
        feature = baseline + noise
        features.append(feature)

    X = np.stack(features, axis=1)

    incident = np.zeros(n_steps, dtype=int)

    starts = []
    attempts = 0
    while len(starts) < incident_count and attempts < incident_count*20:
        attempts += 1
        start = rng.integers(100, n_steps - incident_max_len - 1)
        length =rng.integers(incident_min_len, incident_max_len + 1)

        overlaps = any(not (start + length < s or start > e) for s,e in starts)
        if overlaps:
            continue
        starts.append((start, start + length))

    for start, end in starts:
        incident[start:end] = 1

        ramp_start = max(0, start - 20)
        for j in range(ramp_start, start):
            dist = start - j
            X[j, 0] += max(0, (20-dist) / 20)*2.5
            if n_features > 1:
                X[j, 1] += max(0, (20-dist) / 20)*1.5
            if n_features > 2:
                X[j, 2] += max(0, (20-dist) / 20)*1.0
        
        X[start:end, 0] += 3.0 * rng.normal(0, 0.3, size=end-start)
        if n_features > 1:
            X[start:end, 1] += 2.0 * rng.normal(0, 0.3, size=end-start)
        if n_features > 2:
            X[start:end, 2] -= 1.2 + rng.normal(0, 0.2, size=end-start)

    cols = [f"metric{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df['incident'] = incident
    df['t'] = np.arange(n_steps)

    return df