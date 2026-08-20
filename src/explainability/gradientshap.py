"""
GradientSHAP attribution for LSTM, 1D-CNN, and Transformer models.

What GradientSHAP does
-----------------------
For a given sensor window and its RUL prediction, GradientSHAP
produces a (30, 14) attribution map — one value per cycle per sensor.

  Positive value = this sensor at this cycle pushed the predicted RUL UP
  Negative value = this sensor at this cycle pushed the predicted RUL DOWN
  Near zero      = this sensor at this cycle had little influence

"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from captum.attr import GradientShap


class GradientSHAPExplainer:
    """
    Wraps Captum's GradientShap to produce (seq, features) attribution
    maps for any differentiable PyTorch model.

    """

    def __init__(
        self,
        model:     torch.nn.Module,
        device:    torch.device,
        n_samples: int = 50,
        seed:      int = 42,
    ):
        self.model    = model.to(device)
        self.device   = device
        self.n_samples = n_samples
        self.seed     = seed

        self.model.eval()

        # Initialise Captum's GradientShap wrapper
        # It expects the model to accept (batch, seq, features) and return a scaler or (batch,) output which our models do.
        self.gs = GradientShap(self.model)

    def attribute(self, x:torch.Tensor, target_idx: int = None,):
        """
        Compute GradientSHAP attributions for one input window.
        
        Ensure input is on the right device and requires gradient
    
        """
        x = x.to(self.device).float()

        # Baseline: a tensor of zeros — represents minimum sensor readings
        baseline = torch.zeros_like(x).to(self.device)

        torch.manual_seed(self.seed)

        # Compute attributions
        # n_samples controls how many noisy baselines are averaged
        # additional_forward_args is not needed for our simple models
        attributions = self.gs.attribute(
            inputs     = x,
            baselines  = baseline,
            n_samples  = self.n_samples,
            stdevs     = 0.09,)   # std of Gaussian noise added to baseline
                                 # 0.09 ≈ 9% of normalised range [0,1]

        # attributions shape: (1, 30, 14) — squeeze batch dimension
        return attributions.squeeze(0).detach().cpu().numpy()   # (30, 14)

    def attribute_batch(self,X: np.ndarray,):
        all_attrs = []
        for i in range(len(X)):
            window = torch.tensor(X[i:i+1], dtype=torch.float32)
            attr   = self.attribute(window)
            all_attrs.append(attr)
            if (i+1) % 10 == 0:
                print(f'  GradientSHAP: {i+1}/{len(X)} windows done')
        return np.stack(all_attrs, axis=0)   # (N, 30, 14)

    def plot(
        self,
        attributions:    np.ndarray,
        sensor_names:    list = None,
        title:           str  = 'GradientSHAP Attribution Map',
        figsize:         tuple = (12, 6),
        save_path:       str  = None,
    ):
        """
        Plot a (30, 14) attribution heatmap.

        Colour scale:
          Red   = positive attribution (sensor pushed RUL prediction up)
          Blue  = negative attribution (sensor pushed RUL prediction down)
          White = near-zero (sensor had little influence)

        """
        if sensor_names is None:
            sensor_names = [
                's_2','s_3','s_4','s_7','s_8','s_9','s_11',
                's_12','s_13','s_14','s_15','s_17','s_20','s_21'
            ]

        fig, axes = plt.subplots(1, 2, figsize=figsize,
                                 gridspec_kw={'width_ratios': [3, 1]})

        # Left: attribution heatmap 
        # Transpose so sensors are on y-axis and cycles on x-axis
        attr_T = attributions.T   # (14, 30)

        # Symmetric colour scale centred at zero
        vmax = np.abs(attributions).max()
        vmax = vmax if vmax > 0 else 1.0

        im = axes[0].imshow(
            attr_T,
            cmap   = 'RdBu_r',    # Red=positive, Blue=negative, White=zero
            aspect = 'auto',
            vmin   = -vmax,
            vmax   =  vmax,
        )

        axes[0].set_xlabel('Cycle within window (0=oldest, 29=most recent)',
                           fontsize=10)
        axes[0].set_ylabel('Sensor', fontsize=10)
        axes[0].set_yticks(range(len(sensor_names)))
        axes[0].set_yticklabels(sensor_names, fontsize=8)
        axes[0].set_title(title, fontsize=12)
        plt.colorbar(im, ax=axes[0], label='Attribution (red=↑RUL, blue=↓RUL)')

        # Right: per-sensor total absolute attribution
        # Collapses time dimension shows which sensors mattered most overall
        sensor_importance = np.abs(attributions).mean(axis=0)   # (14,)
        sorted_idx        = np.argsort(sensor_importance)

        colors = ['#d62728' if sensor_importance[i] > np.median(sensor_importance)
                  else '#aec7e8' for i in sorted_idx]

        axes[1].barh(
            range(len(sensor_names)),
            sensor_importance[sorted_idx],
            color=colors,
            edgecolor='white',
        )
        axes[1].set_yticks(range(len(sensor_names)))
        axes[1].set_yticklabels([sensor_names[i] for i in sorted_idx], fontsize=8)
        axes[1].set_xlabel('Mean |attribution|', fontsize=9)
        axes[1].set_title('Sensor importance', fontsize=10)
        axes[1].axvline(np.median(sensor_importance), color='grey',
                        linestyle='--', linewidth=0.8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=120)
            print(f'Saved → {save_path}')

        plt.show()

    def plot_sensor_over_time(
        self,
        attributions: np.ndarray,
        sensor_idx:   int,
        sensor_name:  str = '',
        title:        str = '',
    ):
        """
        Plot how attribution for one sensor varies across the 30 cycles.
        Useful for seeing exactly which cycles of a sensor drove the prediction.

        """
        attr_over_time = attributions[:, sensor_idx]   # (30,)

        fig, ax = plt.subplots(figsize=(10, 3))
        colors  = ['#d62728' if v > 0 else '#1f77b4' for v in attr_over_time]
        ax.bar(range(30), attr_over_time, color=colors, edgecolor='white')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Cycle within window')
        ax.set_ylabel('Attribution')
        ax.set_title(title or f'GradientSHAP: {sensor_name} attribution over 30 cycles')
        plt.tight_layout()
        plt.show()
