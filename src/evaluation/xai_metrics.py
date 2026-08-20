"""
Quantitative evaluation of XAI explanations across three dimensions:

1. FAITHFULNESS  — does removing important features hurt predictions most?
2. STABILITY     — do similar inputs produce similar explanations?
3. COMPLEXITY    — how concentrated/sparse are the attributions?

"""

import numpy as np
import torch
from typing import Callable


# FAITHFULNESS 

def faithfulness_deletion(
    model:       torch.nn.Module,
    X:           np.ndarray,
    attributions: np.ndarray,
    device:      torch.device,
    steps:       int = 14,
):
    
    # Deletion test: progressive feature removal by attribution rank.
    
    model.eval()
    N, T, F = X.shape   # (100, 30, 14)

    # Compute baseline RMSE (no sensors removed)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds_base = model(X_tensor).cpu().numpy()

    # Rank sensors by mean absolute attribution across all windows and cycles
    # sensor_importance shape: (14,)
    sensor_importance = np.abs(attributions).mean(axis=(0, 1))
    ranked_sensors    = np.argsort(sensor_importance)[::-1]   # most → least

    rmse_curve      = []
    sensors_removed = []

    for step in range(steps + 1):
        # Zero out top-step sensors
        X_modified = X.copy()
        for k in range(step):
            sensor_idx = ranked_sensors[k]
            X_modified[:, :, sensor_idx] = 0.0

        X_mod_tensor = torch.tensor(X_modified, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds_mod = model(X_mod_tensor).cpu().numpy()

        # RMSE of prediction change from baseline
        rmse = float(np.sqrt(np.mean((preds_mod - preds_base) ** 2)))
        rmse_curve.append(rmse)
        sensors_removed.append(step)

    # AUC of deletion curve
    # lower = explanation is more faithful
    # High initial RMSE drop = removing important sensors hurts fast
    auc = float(np.trapezoid(rmse_curve, dx=1.0) / steps)

    return {
        'rmse_curve'     : rmse_curve,
        'sensors_removed': sensors_removed,
        'ranked_sensors' : ranked_sensors.tolist(),
        'auc'            : auc,
        'initial_rmse'   : rmse_curve[0],
        'final_rmse'     : rmse_curve[-1],
        'peak_rmse'      : max(rmse_curve),
    }


def faithfulness_random_baseline(
    model:  torch.nn.Module,
    X:      np.ndarray,
    device: torch.device,
    steps:  int = 14,
    seed:   int = 42,
):
    
    # Random deletion baseline — sensors removed in random order.
    model.eval()
    rng = np.random.default_rng(seed)
    random_order = rng.permutation(14)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds_base = model(X_tensor).cpu().numpy()

    rmse_curve = []
    for step in range(steps + 1):
        X_modified = X.copy()
        for k in range(step):
            X_modified[:, :, random_order[k]] = 0.0
        X_mod_tensor = torch.tensor(X_modified, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds_mod = model(X_mod_tensor).cpu().numpy()
        rmse = float(np.sqrt(np.mean((preds_mod - preds_base) ** 2)))
        rmse_curve.append(rmse)

    auc = float(np.trapezoid(rmse_curve, dx=1.0) / steps)
    return {
        'rmse_curve'     : rmse_curve,
        'sensors_removed': list(range(steps + 1)),
        'ranked_sensors' : random_order.tolist(),
        'auc'            : auc,
        'initial_rmse'   : rmse_curve[0],
        'final_rmse'     : rmse_curve[-1],
        'peak_rmse'      : max(rmse_curve),
    }


# STABILITY 

def stability_score(
    explainer_fn:  Callable,
    X:             np.ndarray,
    n_samples:     int   = 20,
    noise_std:     float = 0.05,
    seed:          int   = 42,
):
   
   # Stability (Lipschitz continuity) of explanations.

    
    rng = np.random.default_rng(seed)
    lipschitz_scores = []

    # Use subset of windows for speed
    X_eval = X[:20]

    for i, window in enumerate(X_eval):
        base_window = window[np.newaxis]  # (1, 30, 14)
        base_tensor = torch.tensor(base_window, dtype=torch.float32)

        # Compute base attribution
        base_attr = explainer_fn(base_tensor)   # (30, 14)

        ratios = []
        for _ in range(n_samples):
            # Add small Gaussian noise
            noise = rng.normal(0, noise_std, base_window.shape).astype(np.float32)
            noisy_window = np.clip(base_window + noise, 0, 1)
            noisy_tensor = torch.tensor(noisy_window, dtype=torch.float32)

            # Compute attribution for noisy input
            noisy_attr = explainer_fn(noisy_tensor)  # (30, 14)

            # Lipschitz ratio: ||attr_diff|| / ||input_diff||
            attr_diff  = np.linalg.norm(noisy_attr - base_attr)
            input_diff = np.linalg.norm(noise)

            if input_diff > 1e-8:
                ratios.append(attr_diff / input_diff)

        if ratios:
            lipschitz_scores.append(float(np.mean(ratios)))

        if (i + 1) % 5 == 0:
            print(f'  Stability: {i+1}/{len(X_eval)} windows  '
                  f'running mean={np.mean(lipschitz_scores):.4f}')

    return {
        'lipschitz_scores': lipschitz_scores,
        'mean_lipschitz'  : float(np.mean(lipschitz_scores)),
        'std_lipschitz'   : float(np.std(lipschitz_scores)),
    }


# COMPLEXITY

def complexity_score(
    attributions: np.ndarray,
):
   
    # Complexity of explanations: Gini coefficient of attribution sparsity.

   
    N = len(attributions)
    gini_scores    = []
    top1_fractions = []
    top3_fractions = []

    for i in range(N):
        # Sensor importance: mean |attribution| across 30 cycles → (14,)
        importance = np.abs(attributions[i]).mean(axis=0)
        total = importance.sum()
        if total < 1e-10:
            gini_scores.append(0.0)
            top1_fractions.append(1/14)
            top3_fractions.append(3/14)
            continue

        # Normalise to sum to 1
        importance = importance / total
        n = len(importance)
        sorted_imp = np.sort(importance)   # ascending
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_imp) / (n * sorted_imp.sum())) - (n + 1) / n
        gini = float(np.clip(gini, 0, 1))
        gini_scores.append(gini)

        # Top-k concentration
        sorted_desc = np.sort(importance)[::-1]
        top1_fractions.append(float(sorted_desc[0]))
        top3_fractions.append(float(sorted_desc[:3].sum()))

    return {
        'gini_per_window' : gini_scores,
        'mean_gini'       : float(np.mean(gini_scores)),
        'std_gini'        : float(np.std(gini_scores)),
        'top1_fraction'   : float(np.mean(top1_fractions)),
        'top3_fraction'   : float(np.mean(top3_fractions)),
    }


# SUMMARY TABLE 

def print_xai_evaluation_table(results: dict):
    
    print(f'\n{"="*72}')
    print(f'{"XAI EVALUATION FRAMEWORK — RESULTS":^72}')
    print(f'{"="*72}')
    print(f'{"Model":<14} {"Method":<18} {"Faithful↑":>10} '
          f'{"Stable↓":>9} {"Gini↑":>8} {"Top3%":>8}')
    print(f'{"-"*72}')

    for model_name, methods in results.items():
        for method_name, metrics in methods.items():
            faith  = metrics['faithfulness']['auc']
            stab   = metrics['stability']['mean_lipschitz']
            gini   = metrics['complexity']['mean_gini']
            top3   = metrics['complexity']['top3_fraction'] * 100
            print(f'  {model_name:<12} {method_name:<18} '
                  f'{faith:>10.4f} {stab:>9.4f} {gini:>8.4f} {top3:>7.1f}%')

    print(f'{"="*72}')
    print('Faithful↑ : peak RMSE drop when top sensors removed (higher = more faithful)')
    print('Stable↓   : mean Lipschitz estimate (lower = more stable)')
    print('Gini↑     : attribution sparsity (higher = simpler explanation)')
    print('Top3%     : % of importance held by top 3 sensors (higher = more focused)')
