import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.utils.config import load_config
from src.utils.io import load_pickle, save_json
from src.data.dataset import create_sliding_windows, TimeSeriesWindowDataset
from src.data.preprocessing import temporal_split, transform_windows
from src.models.lstm_classifier import LSTMIncidentPredictor
from src.inference import predict_probabilities
from src.utils.metrics import find_best_threshold, compute_classification_metrics, compute_event_level_metrics
from src.serving.alerting import probabilities_to_alerts

def evaluate():
    config = load_config("configs/default.yaml")

    import pandas as pd
    df = pd.read_csv(config["paths"]["data_path"])

    metric_cols = [c for c in df.columns if c.startswith("metric_")]
    metrics = df[metric_cols].values
    incident = df["incident"].values

    X,y, decision_times = create_sliding_windows(
        metrics=metrics,
        incident=incident,
        window_size=config["data"]["window_size"],
        horizon=config["data"]["horizon"],
    )
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = temporal_split(
        X,y, 
        train_ratio=config["data"]["train_ratio"],
        val_ratio=config["data"]["val_ratio"],
        test_ratio=config["data"]["test_ratio"],
    )

    total = len(X)
    train_end = int(total * config["data"]["train_ratio"])
    val_end = train_end + int(total * config["data"]["val_ratio"])
    decision_times_test = decision_times[val_end:]

    scaler = load_pickle(config["paths"]["scaler_path"])
    X_val = transform_windows(X_val, scaler)
    X_test = transform_windows(X_test, scaler)

    val_ds = TimeSeriesWindowDataset(X_val, y_val)
    test_ds = TimeSeriesWindowDataset(X_test, y_test)

    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMIncidentPredictor(
        input_size = config["model"]["input_size"],
        hidden_size = config["model"]["hidden_size"],
        num_layers = config["model"]["num_layers"],
        dropout = config["model"]["dropout"],
    ).to(device)

    state_dict = torch.load(config["paths"]["model_path"], map_location=device)
    model.load_state_dict(state_dict)

    val_probs = predict_probabilities(model, val_loader, device)
    best_threshold, _ = find_best_threshold(y_val, val_probs)

    test_probs = predict_probabilities(model, test_loader, device)
    clf_metrics = compute_classification_metrics(y_test, test_probs, best_threshold)

    alert_flags = probabilities_to_alerts(test_probs, threshold=best_threshold)
    alert_times = decision_times_test[alert_flags == 1]

    incident_test_sequence = incident[decision_times_test[0]: decision_times_test[-1] + config["data"]["horizon"]]
    event_metrics = compute_event_level_metrics(
        incident_sequence=incident_test_sequence,
        alert_times=alert_times - decision_times_test[0],  
        decision_times=decision_times_test - decision_times_test[0],
    )

    all_metrics = {
        "threshold": best_threshold,
        "classification_metrics": clf_metrics,
        "event_metrics": event_metrics,
    }

    save_json(all_metrics, config["paths"]["metrics_path"])

    plt.figure(figsize=(12,5))
    plt.plot(decision_times_test[:300], test_probs[:300], label="Predicted incident probability")
    plt.plot(
        decision_times_test[:300], 
        y_test[:300],
        label="Ground truth (window label)", 
        alpha=0.7
    )
    plt.axhline(best_threshold, linestyle="--", label=f"Threshold ({best_threshold:.2f})")
    plt.legend()
    plt.xlabel("Time step")
    plt.ylabel("Probability / label")
    plt.title("Predicted alerting on test set")
    plt.tight_layout()
    plt.savefig(config["paths"]["plot_path"], dpi = 150)

    print(json.dumps(all_metrics, indent=2))

if __name__ == "__main__":
    evaluate()