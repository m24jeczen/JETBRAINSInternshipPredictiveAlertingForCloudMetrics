import torch
import torch.nn as nn

class LSTMIncidentPredictor(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_layers:int, dropout: float):
        super().__init__()

        effective_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        output, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        logits = self.classifier(last_hidden).squeeze(-1)
        return logits