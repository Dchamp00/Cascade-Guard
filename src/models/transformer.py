"""
--------------
Transformer Encoder model for RUL regression on NASA CMAPSS.

Key hyperparameters
--------------------
d_model = 64    : dimension of internal representations
nhead   = 4     : number of parallel attention heads
                  Each head learns a different type of relationship.
                  4 heads * 16 dims each = 64 total.
num_layers = 3  : encoder layers stacked
                  More layers = more abstract pattern detection.
dim_feedforward = 256 : width of the FFN inside each encoder layer.
                  Typically 4* d_model (4 * 64 = 256).

Input  : (batch, seq_len=30, input_size=14)
Output : (batch,) — predicted RUL in cycles
"""

import math
import torch
import torch.nn as nn


# Part 1: Positional Encoding 

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build the positional encoding matrix once at init time
        # pe shape: (max_len, d_model)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()   # (max_len, 1)

        # Frequency terms: different frequencies for different dimensions
        # Even dimensions = sin, odd dimensions = cos
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(pos * div)   # even dimensions
        pe[:, 1::2] = torch.cos(pos * div)   # odd dimensions

        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor):
        
        return self.dropout(x + self.pe[:, :x.size(1)])


# Part 2: Transformer Model 

class TransformerModel(nn.Module):

    def __init__(
        self,
        input_size:      int   = 14,    # number of informative sensors
        d_model:         int   = 64,    # internal representation dimension
        nhead:           int   = 4,     # number of attention heads
        num_layers:      int   = 3,     # number of encoder layers
        dim_feedforward: int   = 256,   # FFN hidden dimension (4 × d_model)
        dropout:         float = 0.1,   # dropout throughout
    ):
        super().__init__()
        """
        Input projection 
        The raw sensor values (14 per cycle) need to be projected into the d_model=64 dimensional space that the Transformer works in.
        Every cycle's 14 sensor values get mapped to 64 dimensions.
        Why 64 dimensions?
        Must be divisible by nhead (4). 64/4 = 16 dims per head, Large enough to represent complex sensor relationships, small enough to train quickly on CPU.
        """
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional encoding 
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        """
        Transformer encoder 
        TransformerEncoderLayer is one complete encoder block:
        Multi-Head Self-Attention -> Add & Norm -> FFN -> Add & Norm
        """
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = True,
        )

        # Stack num_layers encoder layers
        # Each layer refines the representations from the previous layer
        # Layer 1: learns basic relationships between sensor channels
        # Layer 2: learns higher-order temporal patterns
        # Layer 3: learns complex degradation signatures
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers = num_layers,
        )

        # Prediction head 
        # Takes the mean-pooled encoder output (64-dim) and maps to RUL
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Store d_model for use in forward pass
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        """
        Shape trace through the network:
        (batch, 30, 14)
          -> input_proj  -> (batch, 30, 64)
          -> pos_enc     -> (batch, 30, 64)  [position info added]
          -> encoder     -> (batch, 30, 64)  [attention applied 3*layers]
          -> mean(dim=1) -> (batch, 64)      [average across 30 cycles]
          -> head        -> (batch, 1)
          -> squeeze     -> (batch,)
        """
        # Step 1: project 14 sensor dims → 64 model dims
        x = self.input_proj(x)              # (batch, 30, 14) → (batch, 30, 64)

        # Step 2: add positional encoding
        x = self.pos_enc(x)                 # (batch, 30, 64) → (batch, 30, 64)

        # Step 3: run through transformer encoder
        x = self.encoder(x)                 # (batch, 30, 64) → (batch, 30, 64)

        # Step 4: mean pooling across the time dimension
        # Average the 30 cycle representations into one global summary
        x = x.mean(dim=1)                   # (batch, 30, 64) → (batch, 64)

        # Step 5: predict RUL from the global summary
        return self.head(x).squeeze(-1)     # (batch, 64) → (batch,)

    def get_attention_weights(self, x: torch.Tensor):
        
        # Extract attention weights from all encoder layers.

        self.eval()
        attn_weights = []

        with torch.no_grad():
            x = self.pos_enc(self.input_proj(x))

            for layer in self.encoder.layers:
                # access the raw multi-head attention module
                # need_weights=True returns attention weights
                # average_attn_weights=False keeps per-head weights
                _, weights = layer.self_attn(
                    x, x, x,
                    need_weights        = True,
                    average_attn_weights = False,
                )
                attn_weights.append(weights.cpu())

                # manually run the rest of the encoder layer
                # so the next layer gets the correct input
                x = layer(x)

        return attn_weights

    def count_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Smoke test 
if __name__ == "__main__":
    model = TransformerModel()
    dummy = torch.randn(32, 30, 14)
    out   = model(dummy)
    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")
    print(f"Parameters   : {model.count_parameters():,}")
    assert out.shape == (32,), f"Expected (32,) got {out.shape}"

    # Test attention weight extraction
    single = torch.randn(1, 30, 14)
    weights = model.get_attention_weights(single)
    print(f"Attention layers : {len(weights)}")
    print(f"Weights per layer: {weights[0].shape}  (batch, heads, seq, seq)")
    print("Transformer model OK ")
