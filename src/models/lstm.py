"""
lstm.py
-----------------------------------------------------
Stacked LSTM model for RUL regression on NASA CMAPSS.
-----------------------------------------------------

Input shape: (batch, 30, 14)
  30 = window size (cycles), 14 = informative sensors

Output: (batch,) — one predicted RUL per window, in cycles.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(
        self,
        input_size:    int   = 14,     # number of informative sensors
        hidden_size:   int   = 128,    # LSTM hidden state dimension
        num_layers:    int   = 2,      # number of stacked LSTM layers
        dropout:       float = 0.2,    # dropout probability
        bidirectional: bool  = False,  # False = simpler, faster on CPU
    ):
        super().__init__()

        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.directions    = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )

        # Prediction head 
        # Takes the final LSTM hidden state and maps it to one number (RUL)
        # LayerNorm → Linear → ReLU → Dropout → Linear → scalar
        
        fc_in = hidden_size * self.directions

        self.head = nn.Sequential(
            nn.LayerNorm(fc_in),
            nn.Linear(fc_in, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) :
        
        lstm_out, _ = self.lstm(x)

        # Take the hidden state at the final time step (cycle 30)
        last_hidden = lstm_out[:, -1, :]

        # Pass through the prediction head
        out = self.head(last_hidden)
        return out.squeeze(-1)

    def count_parameters(self) :
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


#  Smoke test 
if __name__ == "__main__":
    model = LSTMModel()
    dummy = torch.randn(32, 30, 14)   # batch of 32 windows
    out   = model(dummy)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {model.count_parameters():,}")
    assert out.shape == (32,), f"Expected (32,) got {out.shape}"
    print("LSTM model OK ✓")
