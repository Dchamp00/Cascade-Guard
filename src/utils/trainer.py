import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config import (
    DATA_PROC, OUT_MODELS, OUT_RESULTS, TRAIN_CONFIG, SEED
)
from src.evaluation.metrics import all_metrics, print_metrics


#  Reproducibility 

def set_seed(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Data loading 

def load_splits(subset: str, device: torch.device):
    """
    Load preprocessed .npy arrays and wrap them in PyTorch DataLoaders.

    """
    proc = DATA_PROC / subset

    def to_loader(X, y, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(
            ds,
            batch_size  = TRAIN_CONFIG["batch_size"],
            shuffle     = shuffle,
            num_workers = 0,                              # 0 = main process only (safe on Windows)
            pin_memory  = (device.type == "cuda"),
        )

    return (
        to_loader(np.load(proc / "X_train.npy"),
                  np.load(proc / "y_train.npy"), shuffle=True),
        to_loader(np.load(proc / "X_val.npy"),
                  np.load(proc / "y_val.npy")),
        to_loader(np.load(proc / "X_test.npy"),
                  np.load(proc / "y_test.npy")),
        np.load(proc / "y_test.npy"),       # raw numpy for metric computation
    )


#  Model factory 

def get_model(name: str) -> nn.Module:
    if name == "lstm":
        from src.models.lstm import LSTMModel
        return LSTMModel()
    elif name == "cnn1d":
        from src.models.cnn1d import CNN1DModel
        return CNN1DModel()
    elif name == "transformer":
        from src.models.transformer import TransformerModel
        return TransformerModel()
    else:
        raise ValueError(f"Unknown model: '{name}'. Choose lstm, cnn1d, or transformer.")


# Training loop 

def train(model_name: str, subset: str, cfg: dict = None) -> dict:

    if cfg is None:
        cfg = TRAIN_CONFIG

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*55}")
    print(f"  Model   : {model_name.upper()}")
    print(f"  Subset  : {subset}")
    print(f"  Device  : {device}")
    print(f"{'='*55}")

    # Setup 
    tr_loader, va_loader, te_loader, y_test = load_splits(subset, device)

    model     = get_model(model_name).to(device)
    criterion = nn.MSELoss()

    optimiser = torch.optim.Adam(
        model.parameters(),
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],   # L2 regularisation
    )

    # ReduceLROnPlateau — no verbose argument (removed in PyTorch >= 2.2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser,
        mode     = "min",
        patience = cfg["lr_patience"],
        factor   = cfg["lr_factor"],
    )

    print(f"  Parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Train batches : {len(tr_loader)}  |  Val batches : {len(va_loader)}")

    ckpt_path      = OUT_MODELS / f"{model_name}_{subset}_best.pt"
    best_val_rmse  = float("inf")
    patience_count = 0
    log            = []
    t0             = time.time()

    # Epoch loop 
    for epoch in range(1, cfg["epochs"] + 1):

        # Train phase 
        # model.train() enables dropout and batch norm (if present)
        # model.eval()  disables them — important during validation
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in tr_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimiser.zero_grad()           # clear gradients from last step
            predictions = model(X_batch)    # forward pass
            loss = criterion(predictions, y_batch)   # compute MSE loss
            loss.backward()                 # backpropagation

            # Clip gradients BEFORE the weight update
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad"])

            optimiser.step()                # update weights
            train_loss += loss.item() * len(y_batch)

        train_loss /= len(tr_loader.dataset)   # average over all samples

        # Validation phase 
        model.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():   # no gradients needed — saves memory and time
            for X_batch, y_batch in va_loader:
                preds = model(X_batch.to(device)).cpu().numpy()
                val_preds.append(preds)
                val_targets.append(y_batch.numpy())

        val_preds   = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_rmse    = float(np.sqrt(np.mean((val_preds - val_targets) ** 2)))

        scheduler.step(val_rmse)
        lr_now = optimiser.param_groups[0]["lr"]

        log.append({
            "epoch"     : epoch,
            "train_mse" : round(train_loss, 6),
            "val_rmse"  : round(val_rmse, 4),
            "lr"        : lr_now,
        })

        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Ep {epoch:>3}/{cfg['epochs']}  "
                  f"train_mse={train_loss:.4f}  "
                  f"val_rmse={val_rmse:.4f}  "
                  f"lr={lr_now:.2e}  "
                  f"[{elapsed:.0f}s]")

        # Early stopping & checkpointing 
        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse  = val_rmse
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_count += 1
            if patience_count >= cfg["patience"]:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {cfg['patience']} epochs)")
                break

    # Test evaluation 
    # Load the best checkpoint (not the last epoch)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    test_preds = []

    with torch.no_grad():
        for X_batch, _ in te_loader:
            test_preds.append(model(X_batch.to(device)).cpu().numpy())

    test_preds   = np.concatenate(test_preds)
    test_metrics = all_metrics(y_test, test_preds)

    print(f"\n  ── Test Results ({model_name.upper()}, {subset}) ──")
    print_metrics(test_metrics)
    print(f"  Best val RMSE : {best_val_rmse:.4f}")
    print(f"  Total time    : {time.time() - t0:.0f}s")

    # Save results 
    results = {
        "model"         : model_name,
        "subset"        : subset,
        "best_val_rmse" : best_val_rmse,
        "test"          : test_metrics,
        "epochs_run"    : epoch,
        "log"           : log,
    }

    res_path = OUT_RESULTS / f"{model_name}_{subset}_results.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved  → {res_path}")
    print(f"  Checkpoint     → {ckpt_path}")

    return results


# CLI 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training an xpdm model on NASA CMAPSS data."
    )
    parser.add_argument("--model",  default="lstm",
                        choices=["lstm", "cnn1d", "transformer"])
    parser.add_argument("--subset", default="FD001",
                        choices=["FD001","FD002","FD003","FD004","all"])
    parser.add_argument("--epochs", type=int,
                        default=None,
                        help="Override number of epochs from config")
    args = parser.parse_args()

    cfg = dict(TRAIN_CONFIG)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    subsets = ["FD001","FD002","FD003","FD004"] \
              if args.subset == "all" else [args.subset]

    for s in subsets:
        train(args.model, s, cfg)
