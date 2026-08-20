"""
Attention weight visualisation for the Transformer model.

Attention weights answer: WHICH CYCLES mattered most?
GradientSHAP and IG answer: WHICH SENSORS at WHICH CYCLES mattered?

Together they give the complete picture:
  - Attention: "The model focused heavily on cycles 25-30"
  - SHAP/IG:   "In those cycles, s_11 and s_4 drove the prediction"

"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


class AttentionVisualiser:
   # Extracts and visualises attention weights from the Transformer model.

    def __init__(
        self,
        model:  torch.nn.Module,
        device: torch.device,
    ):
        self.model  = model.to(device)
        self.device = device
        self.model.eval()

    def get_weights(self,x: torch.Tensor,):
        
        #Extract attention weights for one window.

        x = x.to(self.device).float()
        weights_list = self.model.get_attention_weights(x)
        # Each element: (1, 4, 30, 30) → squeeze batch → (4, 30, 30)
        return [w.squeeze(0).numpy() for w in weights_list]

    def plot_attention(
        self,
        x:         torch.Tensor,
        rul:       float = None,
        engine_id: int   = None,
        save_path: str   = None,
    ):
        """
        Plot averaged attention weights across all 4 heads for each layer.
        Produces a 1*3 grid — one heatmap per encoder layer.

        The heatmap shows which cycles attended to which other cycles.
        Bright cells = high attention.

        """
        weights_list = self.get_weights(x)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        for i, (ax, w) in enumerate(zip(axes, weights_list)):
            # w shape: (4, 30, 30) average across 4 heads
            avg_attn = w.mean(axis=0)   # (30, 30)

            im = ax.imshow(avg_attn, cmap='Blues', aspect='auto',
                           vmin=0, vmax=avg_attn.max())
            ax.set_title(f'Layer {i+1}', fontsize=11)
            ax.set_xlabel('Attended-to cycle (key)', fontsize=9)
            ax.set_ylabel('Query cycle', fontsize=9)
            ax.tick_params(labelsize=7)
            plt.colorbar(im, ax=ax, fraction=0.04)

            diag_strength = np.diag(avg_attn).mean()
            ax.set_xlabel(
                f'Attended-to cycle  (diagonal={diag_strength:.3f})',
                fontsize=8
            )

        title = 'Transformer Attention Weights (avg across 4 heads)'
        if engine_id is not None:
            title += f' — Engine {engine_id}'
        if rul is not None:
            title += f'  RUL={rul:.0f} cycles'

        fig.suptitle(title, fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=120)
            print(f'Saved -> {save_path}')
        plt.show()

    def plot_head_comparison(
        self,
        x:         torch.Tensor,
        layer_idx: int  = 0,
        rul:       float = None,
        save_path: str   = None,
    ):
        """
        Plot all 4 attention heads from one encoder layer side by side.
        Shows how different heads specialise on different relationships.

        """
        weights_list = self.get_weights(x)
        w = weights_list[layer_idx]   # (4, 30, 30)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        vmax = w.max()

        for head_idx, ax in enumerate(axes):
            im = ax.imshow(w[head_idx], cmap='Blues', aspect='auto',
                           vmin=0, vmax=vmax)
            ax.set_title(f'Head {head_idx+1}', fontsize=11)
            ax.set_xlabel('Key cycle', fontsize=8)
            if head_idx == 0:
                ax.set_ylabel('Query cycle', fontsize=8)
            ax.tick_params(labelsize=7)
            plt.colorbar(im, ax=ax, fraction=0.04)

        title = f'Layer {layer_idx+1} — All 4 Attention Heads'
        if rul is not None:
            title += f'  (RUL={rul:.0f} cycles)'

        fig.suptitle(title, fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=120)
            print(f'Saved -> {save_path}')
        plt.show()

    def plot_attention_vs_degradation(
        self,
        x:            torch.Tensor,
        sensor_array: np.ndarray,
        sensor_idx:   int  = 2,
        sensor_name:  str  = 's_4',
        layer_idx:    int  = 2,
        save_path:    str  = None,
    ):
        """
        Plot attention weights alongside the actual sensor degradation signal
        to visually confirm the model attends to cycles with the most drift.

        Left  : sensor signal over 30 cycles
        Right : which cycles the LAST cycle (cycle 29) attended to most

        """
        weights_list = self.get_weights(x)
        w = weights_list[layer_idx]       # (4, 30, 30)
        avg_attn = w.mean(axis=0)         # (30, 30) averaged across heads

        # Attention FROM cycle 29 (last cycle — the prediction point) TO all other cycles
        attn_from_last = avg_attn[29, :]  # (30,) last cycle's attention distribution

        sensor_signal = sensor_array[:, sensor_idx]   # (30,)

        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

        # Top: sensor signal
        axes[0].plot(range(30), sensor_signal, color='#1f77b4',
                     linewidth=1.8, marker='o', markersize=3)
        axes[0].set_ylabel(f'{sensor_name} (normalised)', fontsize=10)
        axes[0].set_title(f'Sensor signal vs attention from cycle 29 — Layer {layer_idx+1}',
                          fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # Bottom: attention from the last cycle
        axes[1].bar(range(30), attn_from_last, color='#ff7f0e',
                    edgecolor='white', alpha=0.85)
        axes[1].set_xlabel('Cycle within window', fontsize=10)
        axes[1].set_ylabel('Attention weight\n(from cycle 29)', fontsize=9)
        axes[1].grid(True, alpha=0.3)

        # Mark the cycle with maximum attention
        peak_cycle = int(np.argmax(attn_from_last))
        axes[1].axvline(peak_cycle, color='red', linestyle='--',
                        linewidth=1.5, label=f'Peak attention at cycle {peak_cycle}')
        axes[1].legend(fontsize=9)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=120)
            print(f'Saved → {save_path}')
        plt.show()

    def attention_summary(
        self,
        x: torch.Tensor,
    ):
        """
        Compute summary statistics of attention patterns for one window.
        Useful for comparing attention patterns across engines and subsets.
        
        """
        weights_list = self.get_weights(x)
        avgs = [w.mean(axis=0) for w in weights_list]  # [(30,30), (30,30), (30,30)]

        # Self-attention diagonal strength
        mean_diagonal = float(np.mean([np.diag(a).mean() for a in avgs]))

        # Late-cycle bias — fraction of total attention on cycles 20-29
        late_bias = float(np.mean([
            a[:, 20:].sum() / a.sum() for a in avgs
        ]))

        # Agreement between layer 1 and layer 3 (Pearson on flattened maps)
        layer_agreement = float(np.corrcoef(
            avgs[0].flatten(), avgs[2].flatten()
        )[0, 1])

        return {
            'mean_diagonal'   : mean_diagonal,
            'late_bias'       : late_bias,
            'layer_agreement' : layer_agreement,
        }
