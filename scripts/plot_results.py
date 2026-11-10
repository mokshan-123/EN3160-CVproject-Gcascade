"""Plot G-CASCADE recorded baseline metrics without loading weights or retraining.

Run from repository root: python scripts/plot_results.py
Outputs docs/images/*.png and docs/images/figure_manifest.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EPOCHS = ROOT / 'results/baseline/session_metrics.csv'
SUMMARY = ROOT / 'results/baseline/final_evaluation_summary.csv'
OUTPUT = ROOT / 'docs/images'
COUNTS = {'CVC-ClinicDB': 62, 'Kvasir': 100, 'CVC-ColonDB': 380, 'CVC-300': 60}


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rows(p: Path) -> list[dict[str, str]]:
    with p.open('r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    if not EPOCHS.is_file() or not SUMMARY.is_file():
        raise FileNotFoundError('Required baseline CSV files were not found')
    epochs = rows(EPOCHS)
    summary = rows(SUMMARY)
    if len(epochs) != 90 or [int(r['epoch']) for r in epochs] != list(range(1, 91)):
        raise ValueError('Expected exactly 90 consecutive epoch records')
    if {r['dataset'] for r in summary} != set(COUNTS):
        raise ValueError('Unexpected evaluation datasets')
    if any(int(r['image_count']) != COUNTS[r['dataset']] for r in summary):
        raise ValueError('Unexpected evaluation image counts')
    weights = list(COUNTS.values())
    if sum(weights) != 602:
        raise ValueError('Incorrect total image count')
    for r in epochs:
        measured = (62*float(r['clinicdb_dice']) + 100*float(r['kvasir_dice']) +
                    380*float(r['colondb_dice']) + 60*float(r['cvc300_dice'])) / 602
        if abs(measured - float(r['weighted_dice'])) > 2e-6:
            raise ValueError(f"Weighted Dice check failed at epoch {r['epoch']}")
    e = [int(r['epoch']) for r in epochs]
    dice = [float(r['weighted_dice']) for r in epochs]
    best_i = max(range(len(dice)), key=dice.__getitem__)
    if e[best_i] != 19 or abs(dice[best_i] - 0.849586710963455) > 1e-6:
        raise ValueError('Unexpected best epoch / Dice; inspect before plotting')
    OUTPUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(e, dice, linewidth=1.7, label='Epoch validation Dice')
    ax.axvline(19, linestyle='--', linewidth=1, label='Best: epoch 19')
    ax.scatter([19], [dice[best_i]], s=50, zorder=3)
    ax.set(xlabel='Completed epoch', ylabel='Image-weighted validation Dice',
           title='PVT-GCASCADE — Validation Dice over 90 epochs')
    ax.set_xlim(1, 90)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    curve = OUTPUT / 'validation_dice_by_epoch.png'
    fig.savefig(curve, dpi=220)
    plt.close(fig)

    names = list(COUNTS)
    by_name = {r['dataset']: r for r in summary}
    official_dice = [float(by_name[n]['official_dice']) for n in names]
    official_iou = [float(by_name[n]['official_iou']) for n in names]
    weighted = sum(COUNTS[n]*float(by_name[n]['official_dice']) for n in names)/602
    if abs(weighted - 0.849586710963455) > 1e-6:
        raise ValueError('Final evaluation Dice does not match best checkpoint')
    x = list(range(len(names)))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar([v-0.2 for v in x], official_dice, width=0.4, label='Dice')
    ax.bar([v+0.2 for v in x], official_iou, width=0.4, label='IoU')
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.set(ylabel='Official-style validation score', title='PVT-GCASCADE — Segmentation metrics by dataset')
    ax.legend()
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    dataset_plot = OUTPUT / 'validation_dice_iou_by_dataset.png'
    fig.savefig(dataset_plot, dpi=220)
    plt.close(fig)

    manifest = {
        'status': 'generated_from_recorded_metrics',
        'source_csv_sha256': {str(EPOCHS.relative_to(ROOT)): digest(EPOCHS), str(SUMMARY.relative_to(ROOT)): digest(SUMMARY)},
        'figures': {curve.name: digest(curve), dataset_plot.name: digest(dataset_plot)},
        'completed_epochs': len(epochs),
        'best_epoch': 19,
        'best_weighted_validation_dice': weighted,
        'evaluation_images': 602,
        'metric_note': 'Validation datasets were used for checkpoint selection; not independent holdout tests.',
        'training_or_inference_executed': False,
    }
    (OUTPUT / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print('PASS: 90 epochs, four datasets, 602 images verified')
    print('CREATED:', curve.relative_to(ROOT), dataset_plot.relative_to(ROOT), 'and figure_manifest.json')


if __name__ == '__main__':
    main()
