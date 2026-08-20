"""

1D Convolutional Neural Network for RUL regression on NASA CMAPSS.


Input  : (batch, seq_len=30, input_size=14)
Output : (batch,) — predicted RUL in cycles
"""

import torch
import torch.nn as nn


class CNN1DModel(nn.Module):

    def __init__(
        self,
        input_size:  int   = 14,          # number of informative sensors
        channels:    list  = None,        # output channels per conv block
        kernel_size: int   = 3,           # conv filter width in cycles
        dropout:     float = 0.2,         # dropout probability in head
    ):
        super().__init__()

        if channels is None:
            channels = [32, 64, 128]      # default: 3 blocks, doubling channels

        # Convolutional blocks 
        # Each block: Conv1d → BatchNorm → ReLU → Dropout
        # in_ch starts at input_size (14 sensors) and grows through the list
        layers = []
        in_ch  = input_size

        for out_ch in channels:
            layers += [
                # Conv1d expects (batch, channels, seq_len)
                nn.Conv1d(
                    in_channels  = in_ch,
                    out_channels = out_ch,
                    kernel_size  = kernel_size,
                    padding      = kernel_size // 2,  # 'same' padding — keeps seq length
                ),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_ch = out_ch   # output of this block = input of next block

        self.conv = nn.Sequential(*layers)

        # Global average pooling 
        # Collapses (batch, 128, 30) → (batch, 128, 1) → squeeze → (batch, 128)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Prediction head 
        # Same structure as LSTM head: Linear → ReLU → Dropout → Linear → scalar
        self.head = nn.Sequential(
            nn.Linear(channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor):
        # Conv1d expects channels before sequence: (batch, features, seq_len)
        # Our input is (batch, seq_len, features) — permute to swap axes [(batch, 30, 14) → (batch, 14, 30)]
        x = x.permute(0, 2, 1)

        # Pass through conv blocks: (batch, 14, 30) → (batch, 128, 30)
        x = self.conv(x)

        # Global average pool: (batch, 128, 30) → (batch, 128, 1)
        x = self.pool(x)

        # Remove the trailing size-1 dimension: (batch, 128, 1) → (batch, 128)
        x = x.squeeze(-1)

        # Prediction head: (batch, 128) → (batch, 64) → (batch, 1) → (batch,)
        return self.head(x).squeeze(-1)

    def count_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


#  Smoke test 
if __name__ == "__main__":
    model = CNN1DModel()
    dummy = torch.randn(32, 30, 14)   
    out   = model(dummy)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {model.count_parameters():,}")
    assert out.shape == (32,), f"Expected (32,) got {out.shape}"
    print("1D-CNN model OK ")
