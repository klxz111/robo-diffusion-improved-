import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


class MetricsLogger:
    """Training metrics recorder with real-time visualization.

    Records per-step and per-epoch metrics, saves JSON + CSV,
    and generates a loss_and_std.png at the end of each epoch.
    """

    def __init__(self, log_dir="logs/"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.step_data = []
        self.epoch_data = []

    def log_step(self, d):
        """Record a single training step.

        Expected keys: step, loss, lr, vram_mb, step_time_s
        """
        self.step_data.append(d)

    def log_epoch(self, d):
        """Record epoch-level summary.

        Expected keys: epoch, avg_loss, best_loss, patience,
                       pred_std_x, pred_std_y, real_std_x, real_std_y, lr
        """
        self.epoch_data.append(d)

    def save(self):
        """Persist all metrics to JSON and epoch summary to CSV."""
        json_path = os.path.join(self.log_dir, "metrics.json")
        with open(json_path, "w") as f:
            json.dump({"steps": self.step_data, "epochs": self.epoch_data}, f, indent=2)

        csv_path = os.path.join(self.log_dir, "epochs.csv")
        if self.epoch_data:
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=self.epoch_data[0].keys())
                w.writeheader()
                w.writerows(self.epoch_data)

    def flush_if_needed(self):
        """Periodically flush in-memory data to disk to reduce I/O batch overhead."""
        self.save()

    def plot_and_save(self, epoch):
        """Generate and overwrite loss_and_std.png at epoch end.

        4 subplots:
          1. Training Loss (raw + moving average)
          2. Action Std: Pred vs Real (solid=dashed pairs, linear scale)
          3. Learning Rate (single optimizer, scientific notation)
          4. Best Loss & Patience counter
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Training Loss
        ax = axes[0, 0]
        losses = [d["loss"] for d in self.step_data]
        if len(losses) > 10:
            ax.plot(range(len(losses)), losses, alpha=0.15, color="gray", linewidth=0.5)
            window = min(50, max(10, len(losses) // 5))
            avg = [
                sum(losses[i - window : i]) / window for i in range(window, len(losses))
            ]
            ax.plot(range(window, len(losses)), avg, linewidth=1.5, color="steelblue")
        else:
            ax.plot(range(len(losses)), losses, color="steelblue")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)

        # 2. Action Std: Pred vs Real (linear scale)
        ax = axes[0, 1]
        epochs = [d["epoch"] for d in self.epoch_data]
        pred_std_x = [d.get("pred_std_x", 0) for d in self.epoch_data]
        pred_std_y = [d.get("pred_std_y", 0) for d in self.epoch_data]
        real_std_x = [d.get("real_std_x", 0) for d in self.epoch_data]
        real_std_y = [d.get("real_std_y", 0) for d in self.epoch_data]
        ax.plot(
            epochs, pred_std_x, "b-o", label="Pred Std X", markersize=4, linewidth=1.5
        )
        ax.plot(
            epochs, real_std_x, "b--", label="Real Std X", markersize=4, linewidth=1.5
        )
        ax.plot(
            epochs, pred_std_y, "r-o", label="Pred Std Y", markersize=4, linewidth=1.5
        )
        ax.plot(
            epochs, real_std_y, "r--", label="Real Std Y", markersize=4, linewidth=1.5
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Std (pixel space)")
        ax.set_title("Action Std: Pred vs Real")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # 3. Learning Rate (scientific notation)
        ax = axes[1, 0]
        lr_vals = [d.get("lr", 0) for d in self.epoch_data]
        ax.plot(epochs, lr_vals, "g-o", label="LR", markersize=4, linewidth=1.5)
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # 4. Best Loss & Patience
        ax = axes[1, 1]
        best = [d.get("best_loss", 0) for d in self.epoch_data]
        patience = [d.get("patience", 0) for d in self.epoch_data]
        ax.plot(epochs, best, "c-o", label="Best Loss", markersize=4, linewidth=1.5)
        ax2 = ax.twinx()
        ax2.plot(epochs, patience, "orange", label="Patience", alpha=0.6, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Best Loss")
        ax2.set_ylabel("Patience Counter")
        ax.set_title("Best Loss & Early Stopping")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
        ax.grid(True, alpha=0.3)

        # 5. High Action Ratio (X > 450)
        ax = axes[1, 1]
        train_high = [d.get("high_action_ratio", 0) for d in self.step_data]
        if len(train_high) > 0:
            ax.plot(range(len(train_high)), train_high, alpha=0.15, color="magenta", linewidth=0.5)
            window = min(50, max(10, len(train_high) // 5))
            avg = [sum(train_high[i-window:i])/window for i in range(window, len(train_high))]
            ax.plot(range(window, len(train_high)), avg, linewidth=1.5, color="magenta")
        ax.set_xlabel("Step")
        ax.set_ylabel("Ratio")
        ax.set_title("High Action Ratio (X>450)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        path = os.path.join(self.log_dir, "loss_and_std.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Chart saved: {path}")
