"""
CPU/GPU compatible Trainer subclass.

The original Trainer hardcodes torch.cuda.amp.GradScaler(enabled=True) and
torch.autocast(device_type="cuda"). This subclass auto-detects the device
and skips CUDA-specific features when running on CPU.

Also collects per-epoch metrics and saves training curve plots.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from neuron_proofreader.machine_learning.train import Trainer
from neuron_proofreader.utils import ml_util


class FlexibleTrainer(Trainer):
    """Trainer that works on both CPU and CUDA."""

    def __init__(
        self,
        model,
        model_name,
        output_dir,
        device=None,
        lr=1e-3,
        max_epochs=200,
        min_recall=0,
        save_mistake_mips=False,
    ):
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        super().__init__(
            model,
            model_name,
            output_dir,
            device=device,
            lr=lr,
            max_epochs=max_epochs,
            min_recall=min_recall,
            save_mistake_mips=save_mistake_mips,
        )

        # On CPU, disable GradScaler (parent sets enabled=True for CUDA)
        if device == "cpu":
            self.scaler = torch.amp.GradScaler(enabled=False)

        assert isinstance(self.criterion, nn.BCEWithLogitsLoss), (
            f"Expected BCEWithLogitsLoss criterion, got {type(self.criterion)}"
        )

        # Collect per-epoch stats for plotting
        self.history = {"train": [], "val": []}

        # Raw predictions/labels from the last validation epoch (for diagnostics)
        self.last_val_targets = []
        self.last_val_logits = []

    _EXPECTED_STAT_KEYS = {"loss", "f1", "precision", "recall", "accuracy"}

    def validate_step(self, val_dataloader, epoch):
        """Override to store raw logits and targets for diagnostics."""
        y_accum = []
        hat_y_accum = []
        loss_accum = 0

        self.model.eval()
        with torch.no_grad():
            for x, y in val_dataloader:
                hat_y, loss = self.forward_pass(x, y)
                y_accum.extend(ml_util.tensor_to_list(y))
                hat_y_accum.extend(ml_util.tensor_to_list(hat_y))
                loss_accum += float(ml_util.to_cpu(loss))

        # Store for diagnostics (raw logits, pre-sigmoid)
        self.last_val_targets = y_accum
        self.last_val_logits = hat_y_accum

        stats = self.compute_stats(y_accum, hat_y_accum)
        stats["loss"] = loss_accum / max(len(y_accum), 1)
        self.update_tensorboard(stats, epoch, "val_")
        return stats

    def run(self, train_dataloader, val_dataloader):
        """Override to collect per-epoch stats and plot curves after training."""
        exp_name = os.path.basename(os.path.normpath(self.log_dir))
        print("\nExperiment:", exp_name)
        for epoch in range(self.max_epochs):
            train_stats = self.train_step(train_dataloader, epoch)
            val_stats = self.validate_step(val_dataloader, epoch)

            assert isinstance(train_stats, dict), (
                f"train_step returned {type(train_stats)}, expected dict"
            )
            assert isinstance(val_stats, dict), (
                f"validate_step returned {type(val_stats)}, expected dict"
            )
            assert self._EXPECTED_STAT_KEYS.issubset(train_stats.keys()), (
                f"train_stats missing keys: {self._EXPECTED_STAT_KEYS - train_stats.keys()}"
            )
            assert all(isinstance(v, (int, float)) for v in train_stats.values()), (
                f"train_stats contains non-numeric values: "
                f"{[(k, type(v)) for k, v in train_stats.items() if not isinstance(v, (int, float))]}"
            )

            new_best = self.check_model_performance(val_stats, epoch)

            self.history["train"].append(train_stats)
            self.history["val"].append(val_stats)

            print(f"\nEpoch {epoch}: " + ("New Best!" if new_best else " "))
            self.report_stats(train_stats, is_train=True)
            self.report_stats(val_stats, is_train=False)

            self.scheduler.step()

        self.plot_training_curves()

    def forward_pass(self, x, y):
        """Forward pass with device-appropriate autocast.

        On CUDA the parent class uses torch.autocast for mixed precision.
        On CPU we skip autocast; the TensorDict.to() method already handles
        float64->float32 conversion for all contained tensors.
        """
        if self.device == "cpu":
            x = x.to(self.device)
            y = y.to(self.device)
            hat_y = self.model(x)

            assert hat_y.shape == y.shape, (
                f"Model output shape {hat_y.shape} != target shape {y.shape}"
            )
            assert hat_y.dtype == torch.float32 or hat_y.dtype == torch.float64, (
                f"Model output dtype {hat_y.dtype} is not float"
            )
            assert torch.isfinite(hat_y).all(), (
                f"Model output contains NaN/Inf: "
                f"nan={hat_y.isnan().sum().item()}, inf={hat_y.isinf().sum().item()}"
            )

            loss = self.criterion(hat_y, y)

            assert loss.ndim == 0, f"Loss is not scalar, shape={loss.shape}"
            assert torch.isfinite(loss), f"Loss is {loss.item()}"

            return hat_y, loss
        else:
            return super().forward_pass(x, y)

    def plot_training_curves(self):
        """Save training curve plots to the experiment log directory."""
        if not self.history["train"]:
            return

        metrics = list(self.history["train"][0].keys())
        epochs = np.arange(len(self.history["train"]))

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 3.5))
        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            train_vals = [s[metric] for s in self.history["train"]]
            val_vals = [s[metric] for s in self.history["val"]]

            # Filter out NaN for plotting
            train_valid = [(e, v) for e, v in zip(epochs, train_vals) if not np.isnan(v)]
            val_valid = [(e, v) for e, v in zip(epochs, val_vals) if not np.isnan(v)]

            if train_valid:
                ax.plot(*zip(*train_valid), "o-", label="train", markersize=3)
            if val_valid:
                ax.plot(*zip(*val_valid), "o-", label="val", markersize=3)

            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle(os.path.basename(self.log_dir), fontsize=10)
        fig.tight_layout()

        path = os.path.join(self.log_dir, "training_curves.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Training curves saved to {path}")
