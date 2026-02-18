"""
Training and inference diagnostics for the split proofreading pipeline.

Provides label balance reports, prediction score histograms,
ROC/PR curves, and confusion matrices.
"""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
)


def report_label_balance(collection):
    """Print counts of positive/negative proposals after GT matching.

    Parameters
    ----------
    collection : FragmentsDatasetCollection
        Dataset collection with proposals already generated.
    """
    total_pos = 0
    total_neg = 0
    for key, dataset in collection.datasets.items():
        n_proposals = len(dataset.graph.proposals)
        n_pos = len(dataset.graph.gt_accepts)
        n_neg = n_proposals - n_pos
        total_pos += n_pos
        total_neg += n_neg
        print(f"  [{key}] proposals={n_proposals}  positive={n_pos}  negative={n_neg}")

    total = total_pos + total_neg
    if total == 0:
        print("WARNING: No proposals found. Skipping label balance report.")
        return

    ratio = total_pos / total
    print(f"  Total: {total}  positive={total_pos} ({ratio:.1%})  negative={total_neg} ({1 - ratio:.1%})")
    if total_pos == 0:
        print("  WARNING: 0 positive labels. The model has no positive examples to learn from.")
        print("  This means ROC/PR curves will be degenerate and F1 will be NaN.")


def plot_prediction_histogram(y_scores, y_true=None, save_path="prediction_histogram.png"):
    """Plot histogram of model prediction scores.

    Parameters
    ----------
    y_scores : list[float]
        Predicted probabilities in [0, 1].
    y_true : list[float] or None
        Ground truth labels (0 or 1). If provided, histogram is colored
        by class. If None, single-color histogram (inference case).
    save_path : str
        Path to save the PNG.
    """
    if len(y_scores) == 0:
        print("No predictions to plot histogram.")
        return

    scores = np.array(y_scores)
    bins = np.linspace(0, 1, 21)

    fig, ax = plt.subplots(figsize=(6, 4))

    if y_true is not None:
        labels = np.array(y_true)
        neg_scores = scores[labels == 0]
        pos_scores = scores[labels == 1]
        ax.hist(
            [neg_scores, pos_scores],
            bins=bins,
            stacked=True,
            label=[f"Negative (n={len(neg_scores)})", f"Positive (n={len(pos_scores)})"],
            color=["steelblue", "coral"],
            edgecolor="black",
            linewidth=0.5,
        )
        ax.legend()
    else:
        ax.hist(scores, bins=bins, color="steelblue", edgecolor="black", linewidth=0.5)

    ax.axvline(x=0.8, color="red", linestyle="--", linewidth=1.5, label="Accept threshold (0.8)")
    ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=1, label="Midpoint (0.5)")
    ax.legend(fontsize=8)

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.set_title(f"Prediction Score Distribution (n={len(scores)})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Prediction histogram saved to {save_path}")


def plot_roc_pr_curves(y_true, y_scores, save_path="roc_pr_curves.png"):
    """Plot ROC and precision-recall curves.

    Parameters
    ----------
    y_true : list[float]
        Ground truth labels (0 or 1).
    y_scores : list[float]
        Predicted probabilities in [0, 1].
    save_path : str
        Path to save the PNG.
    """
    labels = np.array(y_true, dtype=int)
    scores = np.array(y_scores)

    if len(labels) < 2:
        print("Too few samples for ROC/PR curves. Skipping.")
        return

    n_classes = len(np.unique(labels))
    if n_classes < 2:
        present = int(labels[0])
        print(f"WARNING: Only class {present} present in labels (n={len(labels)}).")
        print("ROC and PR curves require both classes. Generating placeholder plot.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        for ax, name in [(ax1, "ROC Curve"), (ax2, "Precision-Recall Curve")]:
            ax.text(
                0.5, 0.5,
                f"Undefined\n(only class {present} present)",
                ha="center", va="center", fontsize=12, color="red",
            )
            ax.set_title(name)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"ROC/PR placeholder saved to {save_path}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    ax1.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"AUC = {auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(labels, scores)
    ap = average_precision_score(labels, scores)
    ax2.plot(recall, precision, color="coral", linewidth=2, label=f"AP = {ap:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"ROC/PR curves saved to {save_path}")


def print_confusion_matrix(y_true, y_scores, threshold=0.8):
    """Print confusion matrix at the given threshold (and at 0.5).

    Parameters
    ----------
    y_true : list[float]
        Ground truth labels (0 or 1).
    y_scores : list[float]
        Predicted probabilities in [0, 1].
    threshold : float
        Primary operating threshold.
    """
    labels = np.array(y_true, dtype=int)
    scores = np.array(y_scores)

    if len(labels) == 0:
        print("No predictions for confusion matrix.")
        return

    for thresh in [threshold, 0.5]:
        preds = (scores >= thresh).astype(int)
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        ppv = tp / max(tp + fp, 1)
        npv = tn / max(tn + fn, 1)

        print(f"\n  Confusion Matrix (threshold={thresh})")
        print(f"  {'':15s} Pred Neg   Pred Pos")
        print(f"  {'Actual Neg':15s} {tn:>8d}   {fp:>8d}")
        print(f"  {'Actual Pos':15s} {fn:>8d}   {tp:>8d}")
        print(f"  TPR(recall)={tpr:.3f}  FPR={fpr:.3f}  PPV(precision)={ppv:.3f}  NPV={npv:.3f}")
