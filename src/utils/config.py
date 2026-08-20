from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
DATA_RAW    = ROOT / "data" / "raw" / "CMAPSS"
DATA_PROC   = ROOT / "data" / "processed"
OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_RESULTS = ROOT / "outputs" / "results"

# ── CMAPSS column names (no header in raw files) ───────────────────────────
COL_NAMES = (
    ["unit_id", "cycle"]
    + [f"op_{i}" for i in range(1, 4)]
    + [f"s_{i}"  for i in range(1, 22)]
)

# Sensors with near-zero variance — no predictive signal, dropped before modelling
DROP_SENSORS = ["s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]

# 14 sensors retained after variance filtering
INFORMATIVE_SENSORS = [s for s in [f"s_{i}" for i in range(1, 22)]
                       if s not in DROP_SENSORS]

# Human-readable descriptions for plots
SENSOR_DESC = {
    "s_2" : "Total temp — LPC outlet (R)",
    "s_3" : "Total temp — HPC outlet (R)",
    "s_4" : "Total temp — LPT outlet (R)",
    "s_7" : "Total pressure — HPC outlet (psia)",
    "s_8" : "Physical fan speed (rpm)",
    "s_9" : "Physical core speed (rpm)",
    "s_11": "Static pressure — HPC outlet (psia)",
    "s_12": "Fuel flow ratio (pps/psia)",
    "s_13": "Corrected fan speed (rpm)",
    "s_14": "Corrected core speed (rpm)",
    "s_15": "Bypass ratio",
    "s_17": "Bleed enthalpy",
    "s_20": "HPT coolant bleed (lbm/s)",
    "s_21": "LPT coolant bleed (lbm/s)",
}

# ── Preprocessing (used later) ─────────────────────────────────────────────
RUL_CLIP    = 125   # cap RUL at this value (piecewise-linear label)
WINDOW_SIZE = 30    # sliding window length in cycles
VAL_FRAC  = 0.15    # Fraction of training engines witheld for validation
SEED        = 42   

# ── Output directories ────────────────────────────────────────────────────
OUT_MODELS  = ROOT / "outputs" / "models"
OUT_RESULTS = ROOT / "outputs" / "results"
OUT_FIGURES = ROOT / "outputs" / "figures"

for _p in [OUT_MODELS, OUT_RESULTS, OUT_FIGURES]:
    _p.mkdir(parents=True, exist_ok=True)

# ── Training hyperparameters ──────────────────────────────────────────────
TRAIN_CONFIG = dict(
    epochs       = 100,    # max epochs (early stopping usually kicks in before)
    batch_size   = 256,    # windows per gradient update
    lr           = 1e-3,   # initial learning rate for Adam
    weight_decay = 1e-5,   # L2 regularisation strength
    patience     = 15,     # early stopping patience in epochs
    lr_patience  = 7,      # ReduceLROnPlateau patience in epochs
    lr_factor    = 0.5,    # LR multiplied by this when plateau detected
    clip_grad    = 1.0,    # max gradient norm before clipping
)