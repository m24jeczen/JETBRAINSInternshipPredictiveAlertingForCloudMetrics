import torch
import numpy as np

@torch.no_grad()
def predict_probabilities(model, dataloader, device):
    model.eval()
    probs = []

    for X_batch, _ in dataloader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        batch_probs = torch.sigmoid(outputs).cpu().numpy()
        probs.append(batch_probs)

    return np.concatenate(probs)