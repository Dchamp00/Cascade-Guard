"""
Pipeline (in order)

1.  Load raw data via cmapss_loader.load_dataset()
2.  Assign operating-regime cluster (needed for FD002/FD004)
3.  Fit MinMaxScaler on TRAIN only — one scaler per regime
    (FD001/FD003 = 1 regime,  FD002/FD004 = 6 regimes)
4.  Transform both train and test using the fitted scaler(s)
5.  Build sliding windows  =  X shape (N, 30, 14),  y shape (N,)
    - Train : every valid window per engine (stride = 1)
    - Test  : last window only per engine (end-state RUL prediction)
6.  Split train windows into train / val by unit_id
    (all windows from one engine go entirely to one split — no leakage)
7.  Save arrays as .npy and scaler(s) as .pkl

"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from src.utils.config import (
    DATA_PROC, INFORMATIVE_SENSORS, WINDOW_SIZE, RUL_CLIP, VAL_FRAC, SEED
)
from src.data.cmapss_loader import load_dataset


# Step 1: operating-regime clustering 

def _assign_regime(df: pd.DataFrame):
    """
    Add an 'op_regime' column by rounding op_1 to the nearest integer

    FD001/FD003 = all rows get regime 0 (single condition)
    FD002/FD004 = rows distributed across regimes 0-5
    """
    df = df.copy()
    df["op_regime"] = df["op_1"].round(0).astype(int)
    return df


# Step 2: normalisation 

def _normalise(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    subset:   str,
) :
    
    multi_regime = subset in ("FD002", "FD004")
    train_df = train_df.copy()
    test_df  = test_df.copy()
    scalers  = {}

    if multi_regime:
        for regime in sorted(train_df["op_regime"].unique()):
            mask_tr = train_df["op_regime"] == regime
            mask_te = test_df["op_regime"]  == regime
            sc = MinMaxScaler()
            train_df.loc[mask_tr, INFORMATIVE_SENSORS] = sc.fit_transform(
                train_df.loc[mask_tr, INFORMATIVE_SENSORS]
            )
            if mask_te.any():
                test_df.loc[mask_te, INFORMATIVE_SENSORS] = sc.transform(
                    test_df.loc[mask_te, INFORMATIVE_SENSORS]
                )
            scalers[int(regime)] = sc
    else:
        sc = MinMaxScaler()
        train_df[INFORMATIVE_SENSORS] = sc.fit_transform(
            train_df[INFORMATIVE_SENSORS]
        )
        test_df[INFORMATIVE_SENSORS] = sc.transform(
            test_df[INFORMATIVE_SENSORS]
        )
        scalers[0] = sc

    # Persist scalers — needed at inference and XAI time
    scaler_path = DATA_PROC / f"scaler_{subset}.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scalers, f)
    print(f"  Scalers saved  → {scaler_path}")

    return train_df, test_df


# Step 3: assign RUL to test rows 

def _fill_test_rul(test_df: pd.DataFrame, rul_targets: np.ndarray):
    """
    Attach RUL values to the test DataFrame.

    NASA provides one ground-truth (gt) RUL per test engine (the true RUL at
    the final recorded cycle). We count backwards from that value so
    every row in the test engine's trajectory gets a plausible RUL.
    """
    test_df  = test_df.copy().reset_index(drop=True)
    unit_ids = sorted(test_df["unit_id"].unique())
    rul_map  = dict(zip(unit_ids, rul_targets))

    # Build RUL column row by row
    rul_col = np.zeros(len(test_df), dtype=np.float32)

    for uid, gt in rul_map.items():
        mask     = test_df["unit_id"] == uid
        cycles   = test_df.loc[mask, "cycle"]
        max_cycle = cycles.max()
        rul_vals  = gt + (max_cycle - cycles)
        rul_vals  = rul_vals.clip(upper=RUL_CLIP)
        rul_col[mask.values] = rul_vals.values

    test_df["RUL"] = rul_col
    return test_df

# Step 4: sliding windows 

def _make_windows(
    df: pd.DataFrame,
    window: int  = WINDOW_SIZE,
    is_test: bool = False,
):

    #Convert a per-row sensor DataFrame into fixed-length windows.

    X_list, y_list = [], []

    for uid, grp in df.groupby("unit_id"):
        grp  = grp.sort_values("cycle").reset_index(drop=True)
        vals = grp[INFORMATIVE_SENSORS].values.astype(np.float32)
        rul  = grp["RUL"].values.astype(np.float32)

        if len(vals) < window:
            # Engine shorter than window length — skip silently
            continue

        if is_test:
            X_list.append(vals[-window:])          # last window only
            y_list.append(rul[-1])                 # ground-truth RUL
        else:
            for start in range(len(vals) - window + 1):
                X_list.append(vals[start : start + window])
                y_list.append(rul[start + window - 1])

    X = np.array(X_list, dtype=np.float32)   # (N, 30, 14)(N = Total no. of windows across all engines, 30 = cycles, 14 = sensors)
    y = np.array(y_list,  dtype=np.float32)   # (N,)
    return X, y


#  Step 5: train / val split by unit_id 

def _split_train_val(
    X:       np.ndarray,
    y:       np.ndarray,
    df:      pd.DataFrame,
    val_frac: float = VAL_FRAC,
    seed:     int   = SEED,
):
   
    #Split windows into train and validation sets by engine unit_id.

    rng      = np.random.default_rng(seed)
    unit_ids = df["unit_id"].unique().copy()
    rng.shuffle(unit_ids)

    n_val     = max(1, int(len(unit_ids) * val_frac))
    val_units = set(unit_ids[:n_val].tolist())

    # Rebuild which unit each window belongs to (same order as _make_windows)
    uid_per_window = []
    for uid, grp in df.groupby("unit_id"):
        grp    = grp.sort_values("cycle")
        n_wins = max(0, len(grp) - WINDOW_SIZE + 1)
        uid_per_window.extend([uid] * n_wins)
    uid_arr = np.array(uid_per_window)

    tr_mask = ~np.isin(uid_arr, list(val_units))
    va_mask =  np.isin(uid_arr, list(val_units))

    return X[tr_mask], y[tr_mask], X[va_mask], y[va_mask]


#  Main pipeline 

def preprocess(subset: str) :
    """Run the full preprocessing pipeline for one CMAPSS subset."""
    print(f"\n{'='*55}")
    print(f"  Preprocessing {subset}")
    print(f"{'='*55}")

    # 1. Load raw data
    train_df, test_df, rul_targets = load_dataset(subset)

    # 2. Assign operating regime
    train_df = _assign_regime(train_df)
    test_df  = _assign_regime(test_df)

    # 3. Fill test RUL before normalisation
    test_df = _fill_test_rul(test_df, rul_targets)

    # 4. Normalise (fit on train, apply to both)
    train_df, test_df = _normalise(train_df, test_df, subset)

    # 5. Build sliding windows
    X_all,  y_all  = _make_windows(train_df, is_test=False)
    X_test, y_test = _make_windows(test_df,  is_test=True)

    # 6. Train / val split (no leakage)
    X_tr, y_tr, X_va, y_va = _split_train_val(X_all, y_all, train_df)

    # 7. Save
    out = DATA_PROC / subset
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "X_train.npy", X_tr)
    np.save(out / "y_train.npy", y_tr)
    np.save(out / "X_val.npy",   X_va)
    np.save(out / "y_val.npy",   y_va)
    np.save(out / "X_test.npy",  X_test)
    np.save(out / "y_test.npy",  y_test)

    print(f"\n  Train windows  : {X_tr.shape}")
    print(f"  Val windows    : {X_va.shape}")
    print(f"  Test windows   : {X_test.shape}")
    print(f"  y_train range  : [{y_tr.min():.1f}, {y_tr.max():.1f}]")
    print(f"  y_val range    : [{y_va.min():.1f}, {y_va.max():.1f}]")
    print(f"  y_test range   : [{y_test.min():.1f}, {y_test.max():.1f}]")
    print(f"  Arrays saved   → {out}/")
    print(f"  Done ✓")


# CLI 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess NASA CMAPSS data into sliding-window arrays."
    )
    parser.add_argument(
        "--subset",
        default="FD001",
        choices=["FD001", "FD002", "FD003", "FD004", "all"],
        help="Which CMAPSS subset to preprocess (default: FD001)",
    )
    args = parser.parse_args()

    subsets = ["FD001", "FD002", "FD003", "FD004"] \
              if args.subset == "all" else [args.subset]

    for s in subsets:
        preprocess(s)

    print("\nAll preprocessing complete.")
