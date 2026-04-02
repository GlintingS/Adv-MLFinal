from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_confusion_matrix(conf_matrix: list[list[int]], save_path: str | Path) -> Path:
    """Create and save a confusion matrix figure."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        plt.figure(figsize=(5, 4))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    except Exception:
        logger.exception("Failed to save confusion matrix to %s", save_path)
        raise
    logger.info("Confusion matrix saved to %s", save_path)
    return save_path


def plot_loss_curve(loss_values: list[float], save_path: str | Path) -> Path:
    """Create and save an MLP training loss curve."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        plt.figure(figsize=(7, 4))
        plt.plot(loss_values, color="tab:blue")
        plt.title("MLP Loss Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    except Exception:
        logger.exception("Failed to save loss curve to %s", save_path)
        raise
    logger.info("Loss curve saved to %s", save_path)
    return save_path
