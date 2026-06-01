from __future__ import annotations

import torch
from torch import nn


class RSSILSTM(nn.Module):
    def __init__(
        self,
        num_fog_nodes: int,
        history_len: int,
        pred_horizon: int,
        use_delta: bool = True,
        hidden_size_1: int = 64,
        hidden_size_2: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_fog_nodes = int(num_fog_nodes)
        self.history_len = int(history_len)
        self.pred_horizon = int(pred_horizon)
        self.use_delta = bool(use_delta)
        input_dim = self.num_fog_nodes * (2 if self.use_delta else 1)

        self.lstm1 = nn.LSTM(input_dim, hidden_size_1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size_2, 64),
            nn.ReLU(),
            nn.Linear(64, self.pred_horizon * self.num_fog_nodes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.lstm1(x)
        sequence = self.dropout(sequence)
        _, (hidden, _) = self.lstm2(sequence)
        output = self.head(hidden[-1])
        return output.reshape(x.shape[0], self.pred_horizon, self.num_fog_nodes)
