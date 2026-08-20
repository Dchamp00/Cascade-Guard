# Evaluation metrics for RUL regression.

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) :
    """
    Root Mean Squared Error.
    e.g. RMSE=15 means on average the model is off by 15 cycles.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) :
    """
    Mean Absolute Error.
    Also in cycles. More robust to outliers than RMSE.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) :
    """
    Coefficient of determination (R-squared).
    1.0 = perfect, 0.0 = predicting the mean, negative = worse than mean.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    NASA asymmetric prognostics score (Saxena et al., 2008).

    d = y_pred - y_true
      positive d = early prediction (model overestimates remaining life)
                   -> heavy penalty (exp with divisor 10)
      negative d = late prediction  (model underestimates remaining life)
                   -> mild penalty  (exp with divisor 13)

    Lower is better. Perfect predictions give score = 0.
    """
    d = y_pred - y_true
    scores = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1,   # late prediction — mild penalty
        np.exp( d / 10.0) - 1    # early prediction — heavy penalty
    )
    return float(np.sum(scores))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute all four metrics and return as a dictionary.
    Used by the trainer to save results to JSON.
    """
    return {
        "RMSE"       : rmse(y_true, y_pred),
        "MAE"        : mae(y_true, y_pred),
        "R2"         : r2(y_true, y_pred),
        "NASA_Score" : nasa_score(y_true, y_pred),
    }


def print_metrics(m: dict, label: str = ""):
    """Pretty-print a metrics dictionary."""
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}RMSE={m['RMSE']:.3f}  "
          f"MAE={m['MAE']:.3f}  "
          f"R²={m['R2']:.4f}  "
          f"NASA={m['NASA_Score']:.1f}")
