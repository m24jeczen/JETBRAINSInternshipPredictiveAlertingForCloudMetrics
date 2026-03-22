import os
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.seed import set_seed
from src.utils.config import load_config
from src.utils.io import ensure_dir, save_pickle
from src.data.synthetic import generate_synthetic_cloud_metrcis
from src.data.dataset import create_sliding_windows, TimeSeriesWindowDataset
from src.data.preprocessing import temporal_split, fit_scaler_on_train, transform_windows
from src.models.lstm_classifier import LSTMIncidentPredictor
from src.inference import predict_probabilities
from src.utils.metrics import find_best_threshold, compute_classification_metrics

def train():
    config = load_config("configs/default.yaml")
    set_seed(config["seed"])

    df = generate_synthetic_cloud_metrcis(
        n_steps=config["data"]["n_steps"],
        n_features=config["data"]["n_features"],
        incident_count=config["data"]["incident_count"],
        incident_min_len=config["data"]["incident_min_len"],
        incident_max_len=config["data"]["incident_max_len"],
        seed=config["seed"],
    )

    ensure_dir(config["paths"]["data_path"])
    df.to_csv(config["paths"]["data_path"], index=False)

    metrics_cols = [c for c in df.columns if c.startswith("metric_")]
    metrics = df[metrics_cols].values
    incident = df["incident"].values

    X,y,decision_times = create_sliding_windows(
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


    scaler = fit_scaler_on_train(X_train)
    X_train = transform_windows(X_train, scaler)
    X_val = transform_windows(X_val, scaler)
    X_test = transform_windows(X_test, scaler)

    save_pickle(scaler, config["paths"]["scaler_path"])

    train_ds = TimeSeriesWindowDataset(X_train, y_train)
    val_ds = TimeSeriesWindowDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMIncidentPredictor(
        input_size = config["model"]["input_size"],
        hidden_size = config["model"]["hidden_size"],
        num_layers = config["model"]["num_layers"],
        dropout = config["model"]["dropout"],
    ).to(device)

    pos_weight = torch.tensor(
        [float(config["training"]["positive_class_weight"])], device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits= model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_probs = predict_probabilities(model, val_loader, device)
        best_thr, val_f1 = find_best_threshold(y_val, val_probs)

        print(
            f"Epoch {epoch+1}/{config['training']['epochs']} | "
            f"train Loss: {train_loss/len(train_loader):.4f} | "
            f"val best f1: {val_f1:.4f} | val best thr={best_thr:.4f}" 
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()
        
            os.makedirs("results", exist_ok=True)
            torch.save(best_state, config["paths"]["model_path"])
            print(f"Best model saved to {config['paths']['model_path']}")

if __name__ == "__main__":
    train()