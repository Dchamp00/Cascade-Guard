"""
Integrated Gradients (IG) attribution for LSTM, 1D-CNN, and Transformer.

What makes IG different from GradientSHAP

IG satisfies the COMPLETENESS AXIOM:

    Σ IG(xᵢ) = f(x) - f(x')

    Sum of all attributions = prediction for actual input - prediction for baseline input

"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients


class IGExplainer:
    """
    Wraps Captum's IntegratedGradients to produce (seq, features)
    attribution maps with convergence verification.
    """

    def __init__(
        self,
        model:   torch.nn.Module,
        device:  torch.device,
        n_steps: int = 50,
    ):
        self.model   = model.to(device)
        self.device  = device
        self.n_steps = n_steps

        self.model.eval()

        # Initialise Captum's IntegratedGradients
        self.ig = IntegratedGradients(self.model)

    def attribute(self,x: torch.Tensor,):
        
        # Compute Integrated Gradients attributions for one window.

        x = x.to(self.device).float()

        # Baseline: zeros — same reasoning as GradientSHAP
        baseline = torch.zeros_like(x).to(self.device)

        # Compute IG attributions
       
        attributions, delta = self.ig.attribute(
            inputs                  = x,
            baselines               = baseline,
            n_steps                 = self.n_steps,
            method                  = 'gausslegendre',
            # "gausslegendre" uses Gauss-Legendre quadrature to choose the integration points, more accurate than uniform spacing for the same number of steps
            return_convergence_delta = True,
            internal_batch_size     = 10,
        )

        attr_np  = attributions.squeeze(0).detach().cpu().numpy()  # (30, 14)
        delta_np = float(delta.abs().mean().item())

        return attr_np, delta_np

    def attribute_batch(self,X: np.ndarray,verbose: bool = True,):
        
        # Compute IG attributions for a batch of windows.

        all_attrs  = []
        all_deltas = []

        for i in range(len(X)):
            window = torch.tensor(X[i:i+1], dtype=torch.float32)
            attr, delta = self.attribute(window)
            all_attrs.append(attr)
            all_deltas.append(delta)
            if verbose and (i+1) % 10 == 0:
                print(f'  IG: {i+1}/{len(X)} done  '
                      f'avg delta={np.mean(all_deltas):.4f}')

        return np.stack(all_attrs, axis=0), np.array(all_deltas)

    def plot(
        self,
        attributions: np.ndarray,
        delta:        float = None,
        sensor_names: list  = None,
        title:        str   = 'Integrated Gradients Attribution Map',
        figsize:      tuple = (12, 6),
        save_path:    str   = None,
    ):
        if sensor_names is None:
            sensor_names = [
                's_2','s_3','s_4','s_7','s_8','s_9','s_11',
                's_12','s_13','s_14','s_15','s_17','s_20','s_21'
            ]

        full_title = title
        if delta is not None:
            full_title += f'  (convergence δ={delta:.4f})'

        fig, axes = plt.subplots(1, 2, figsize=figsize,
                                 gridspec_kw={'width_ratios': [3, 1]})

        attr_T = attributions.T   # (14, 30)
        vmax   = np.abs(attributions).max()
        vmax   = vmax if vmax > 0 else 1.0

        im = axes[0].imshow(
            attr_T,
            cmap='RdBu_r',
            aspect='auto',
            vmin=-vmax,
            vmax= vmax,
        )
        axes[0].set_xlabel('Cycle within window', fontsize=10)
        axes[0].set_ylabel('Sensor', fontsize=10)
        axes[0].set_yticks(range(len(sensor_names)))
        axes[0].set_yticklabels(sensor_names, fontsize=8)
        axes[0].set_title(full_title, fontsize=11)
        plt.colorbar(im, ax=axes[0], label='Attribution')

        # Sensor importance bar chart
        importance = np.abs(attributions).mean(axis=0)
        sorted_idx = np.argsort(importance)
        colors     = ['#9467bd' if importance[i] > np.median(importance)
                      else '#c5b0d5' for i in sorted_idx]

        axes[1].barh(range(len(sensor_names)), importance[sorted_idx],
                     color=colors, edgecolor='white')
        axes[1].set_yticks(range(len(sensor_names)))
        axes[1].set_yticklabels([sensor_names[i] for i in sorted_idx], fontsize=8)
        axes[1].set_xlabel('Mean |attribution|', fontsize=9)
        axes[1].set_title('Sensor importance', fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=120)
            print(f'Saved → {save_path}')
        plt.show()

    def compare_with_shap(
        self,
        ig_attr:   np.ndarray,
        shap_attr: np.ndarray,
        sensor_names: list = None,
        title:     str = 'IG vs GradientSHAP — Sensor Importance Comparison',
        save_path: str = None,
    ):
        """
        Side-by-side comparison of IG and GradientSHAP sensor importance.
        If both methods agree on which sensors matter, the explanation
        is more trustworthy (convergent validity).
        """
        if sensor_names is None:
            sensor_names = [
                's_2','s_3','s_4','s_7','s_8','s_9','s_11',
                's_12','s_13','s_14','s_15','s_17','s_20','s_21'
            ]

        ig_imp   = np.abs(ig_attr).mean(axis=0)    # (14,)
        shap_imp = np.abs(shap_attr).mean(axis=0)  # (14,)

        # Normalise to [0,1] for fair comparison
        ig_imp   = ig_imp   / ig_imp.max()
        shap_imp = shap_imp / shap_imp.max()

        # Sort by IG importance
        sorted_idx = np.argsort(ig_imp)

        fig, ax = plt.subplots(figsize=(9, 5))
        y = np.arange(len(sensor_names))

        ax.barh(y - 0.2, ig_imp[sorted_idx],   0.35, label='Integrated Gradients',
                color='#9467bd', alpha=0.85, edgecolor='white')
        ax.barh(y + 0.2, shap_imp[sorted_idx], 0.35, label='GradientSHAP',
                color='#d62728', alpha=0.85, edgecolor='white')

        ax.set_yticks(y)
        ax.set_yticklabels([sensor_names[i] for i in sorted_idx], fontsize=9)
        ax.set_xlabel('Normalised importance (0–1)', fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.axvline(0.5, color='grey', linestyle='--', linewidth=0.8,
                   label='Midpoint')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=120)
            print(f'Saved → {save_path}')
        plt.show()

        # Compute Spearman rank correlation between the two methods
        from scipy.stats import spearmanr
        corr, pval = spearmanr(ig_imp, shap_imp)
        print(f'\nSpearman rank correlation between IG and SHAP importances:')
        print(f'  r = {corr:.3f}  (p = {pval:.4f})')
        print(f'  {"Strong agreement" if corr > 0.8 else "Moderate agreement" if corr > 0.5 else "Weak agreement"} between methods')
