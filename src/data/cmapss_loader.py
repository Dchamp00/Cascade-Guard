import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.config import (
    DATA_RAW, COL_NAMES, DROP_SENSORS, RUL_CLIP
)


def load_raw(subset: str, split: str):
    path = DATA_RAW / f"{split}_{subset}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"\nRaw file not found: {path}\n"
            "Place all 14 .txt files in data/raw/CMAPSS/"
        )
    # engine='python' handles trailing whitespace cleanly
    df = pd.read_csv(path, sep=r"\s+", header=None, engine='python')

    # Take only the first 26 columns — ignore any trailing empty ones
    df = df.iloc[:, :26]
    df.columns = COL_NAMES
    float_cols = [c for c in COL_NAMES if c not in ('unit_id', 'cycle')]
    df[float_cols] = df[float_cols].astype(np.float32)
    df["subset"] = subset
    return df


def compute_rul(df: pd.DataFrame, rul_clip: int = RUL_CLIP) :
    """
    Add a 'RUL' column to a training DataFrame.

    RUL = max_cycle_for_this_unit - current_cycle, clipped at rul_clip.

    Why clip? Early in an engine's life all sensors look identical
    regardless of the true RUL. Capping at 125 removes that flat
    uninformative plateau and forces the model to focus on the
    degradation window where sensor drift is actually visible.

    """
    df = df.copy()
    max_cycles = (
        df.groupby("unit_id")["cycle"]
        .max()
        .rename("max_cycle")
    )
    df = df.merge(max_cycles, on="unit_id")
    df["RUL"] = (df["max_cycle"] - df["cycle"]).clip(upper=rul_clip)
    df.drop(columns="max_cycle", inplace=True)
    return df


def load_rul_targets(subset: str):
    """
    Load the ground-truth RUL file for the test split.

    NASA provides one value per test engine — the true RUL at the
    last recorded cycle of each test engine's trajectory.

    """
    path = DATA_RAW / f"RUL_{subset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"RUL target file not found: {path}")
    return pd.read_csv(path, header=None).values.flatten().astype(np.float32)


def load_dataset(subset: str):
    """
    Convenience wrapper: load and lightly clean train + test for one subset.

    """
    train_df = compute_rul(load_raw(subset, "train"))
    test_df  = load_raw(subset, "test")

    # Drop near-zero-variance sensors from both splits
    for df in [train_df, test_df]:
        cols_to_drop = [c for c in DROP_SENSORS if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)

    rul_targets = load_rul_targets(subset)
    return train_df, test_df, rul_targets


# Quick smoke test
if __name__ == "__main__":
    import sys
    subset = sys.argv[1] if len(sys.argv) > 1 else "FD001"
    tr, te, rul = load_dataset(subset)
    print(f"\n{subset} — train shape : {tr.shape}")
    print(f"{subset} — test  shape : {te.shape}")
    print(f"{subset} — RUL targets : {rul.shape}  min={rul.min():.0f}  max={rul.max():.0f}")
    print(f"\nTrain columns : {list(tr.columns)}")
    print(f"RUL range     : [{tr['RUL'].min():.0f}, {tr['RUL'].max():.0f}]")
    print("\ncmapss_loader OK")
