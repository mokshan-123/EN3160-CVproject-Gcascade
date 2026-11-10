"""Generate reproducible EN3160 plots using the team's committed baseline CSV files.

Run from the team repository root:
    python scripts/generate_project_figures.py
Outputs only PNG figures under docs/images, never model weights or data.
"""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "baseline"
OUT = ROOT / "docs" / "images"
EPOCH_CSV = RESULTS / "session_metrics.csv"
SUMMARY_CSV = RESULTS / "final_evaluation_summary.csv"


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required results CSV is missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    rows = read_csv(EPOCH_CSV)
    scores = read_csv(SUMMARY_CSV)
    if not rows or not scores:
        raise ValueError("Results CSV is empty")
    epochs = [int(r["epoch"]) for r in rows]
    dice = [float(r["weighted_dice"]) for r in rows]
    if epochs != list(range(1, 91)):
        raise ValueError(f"Expected epochs 1-90 in order, got {len(epochs)} rows")
    best_idx = max(range(len(dice)), key=lambda i: dice[i])
    if epochs[best_idx] != 19:
        raise ValueError("Unexpected best epoch; review training CSV")
    if abs(dice[best_idx] - 0.849586710963455) > 1e-9:
        raise ValueError("Best Dice differs from recorded checkpoint")
    order = ["CVC-ClinicDB", "Kvasir", "CVC-ColonDB", "CVC-300"]
    by_name = {r["dataset"]: r for r in scores}
    if set(by_name) != set(order):
        raise ValueError("Unexpected dataset names in evaluation summary")
    if sum(int(by_name[n]["image_count"]) for n in order) != 602:
        raise ValueError("Expected exactly 602 evaluation images")
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(epochs, dice, linewidth=1.6, label="Image-weighted validation Dice")
    ax.scatter([19], [dice[best_idx]], marker="o", s=60, zorder=5)
    ax.annotate(f"Best: epoch 19, Dice={dice[best_idx]:.4f}",
                xy=(19, dice[best_idx]), xytext=(28, .78),
                arrowprops=dict(arrowstyle="->"))
    ax.set(xlabel="Completed training epoch", ylabel="Weighted validation Dice",
           title="PVT-GCASCADE: validation Dice during 90 training epochs")
    ax.set_xlim(1, 90)
    ax.set_ylim(0.7, 0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "validation_dice_by_epoch.png", dpi=180)
    plt.close(fig)

    d = [float(by_name[n]["official_dice"]) for n in order]
    j = [float(by_name[n]["official_iou"]) for n in order]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = list(range(4))
    width = 0.36
    ax.bar([i-width/2 for i in x], d, width, label="Official Dice")
    ax.bar([i+width/2 for i in x], j, width, label="Official IoU")
    ax.set_xticks(x, order)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean segmentation score")
    ax.set_title("Best checkpoint (epoch 19): validation performance by dataset")
    ax.legend()
    for index, val in enumerate(d):
        ax.text(index-width/2, val+0.015, f"{val:.3f}", ha="center", fontsize=9)
    for index, val in enumerate(j):
        ax.text(index+width/2, val+0.015, f"{val:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "validation_dice_iou_by_dataset.png", dpi=180)
    plt.close(fig)
    print("PASS: 90 epochs and 602 images validated")
    for name in ("validation_dice_by_epoch.png", "validation_dice_iou_by_dataset.png"):
        print(f"CREATED: {(OUT / name).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
