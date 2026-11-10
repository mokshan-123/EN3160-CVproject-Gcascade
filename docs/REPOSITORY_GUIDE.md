# G-CASCADE repository and final demonstration guide

## Scope and attribution

This is an EN3160 coursework reproduction of the PVTv2-B2 + additive G-CASCADE polyp segmentation pipeline from Rahman and Marculescu (WACV 2024). Network architecture is the original authors' contribution. This repository's verified engineering work includes compatibility fixes, resumable training, evaluation and visualization, figures, and a local Streamlit app. Image-corruption experiments belong to the proposed team extension; report them as completed only after their outputs and methodology are verified.

## Local environment and assets

Use the verified `gcascade` Conda environment (training used Python 3.10.21, PyTorch 2.2.2+cu121, 6 GB RTX 4050 laptop GPU). Dataset images/masks and model checkpoints are deliberately not in Git. The demo requires the trained `*-best.pth` checkpoint and `data/polyp/TestDataset` or explicit paths set in the Streamlit sidebar.

## Code map for viva

| File or directory | What to explain |
| --- | --- |
| `lib/networks.py`, `lib/decoders.py`, `lib/gcn_lib/` | PVTv2-B2 feature extraction, upsampling, graph convolution and multi-stage predictions |
| `train_polyp.py` | Three-scale training, 15 nonempty combinations of four decoder predictions, combined structure loss, validation and best-checkpoint selection |
| `utils/training_checkpoint.py`, `utils/training_sessions.py`, `scripts/run_polyp_session.ps1` | Full-state resumption and planned 5-epoch sessions |
| `utils/polyp_evaluation.py`, `scripts/evaluate_polyp.py` | Input processing, predicted masks, per-image Dice/IoU and dataset reports |
| `scripts/demo_polyp.py`, `scripts/web_demo_polyp.py` | One-image inference and Streamlit interface with optional ground truth |
| `results/baseline/` | Saved epoch metrics, best-checkpoint metadata and verified dataset summary |
| `scripts/plot_results.py`, `docs/images/` | Plots derived from CSVs, not recreated from memory |
| `tests/` | Unit tests supporting behavior and interface verification |

## Verified training and evaluation

- Completed training: 90 of 199 planned epochs in 18 sessions; best checkpoint epoch 19.
- Best image-count-weighted validation Dice: 0.849586710963455.
- Evaluation: 602 validation images across CVC-ClinicDB (62), Kvasir (100), CVC-ColonDB (380), and CVC-300 (60).

| Dataset | Images | Dice | IoU |
| --- | ---: | ---: | ---: |
| CVC-ClinicDB | 62 | 0.928477 | 0.880758 |
| Kvasir | 100 | 0.916251 | 0.864641 |
| CVC-ColonDB | 380 | 0.811656 | 0.729943 |
| CVC-300 | 60 | 0.897185 | 0.831153 |
| Image-weighted overall | 602 | 0.849587 | 0.777938 |

**Scientific limitation:** The datasets were used to select the best checkpoint and evaluate it. These are validation scores, not independent hold-out test results. The full 199-epoch schedule was intentionally not completed.

## Presentation-safe commands

From the repository root:

```powershell
conda activate gcascade
python -m unittest -v tests.test_training_sessions tests.test_polyp_evaluation tests.test_demo_polyp
python scripts/plot_results.py
python -m streamlit run scripts/web_demo_polyp.py
```

In the app, point the sidebar at the local checkpoint and validation root. Select `CVC-ClinicDB`, then `171.png` and Run segmentation; show input, predicted mask, overlay, ground truth and per-image Dice/IoU. A previously verified example achieved Dice ≈ 0.9757, IoU ≈ 0.9526. If an uploaded image has no matching ground-truth annotation, do not claim its Dice/IoU.

## What not to claim

- Do not claim a new architecture or completed robustness research without experiments.
- Do not call these validation scores independent external test metrics.
- Do not confuse the final best-model checkpoint with the epoch-90 resumption checkpoint.
- Do not imply a complete 199-epoch reproduction, five-seed paper comparison, or that large weights are available through Git.

## Suggested 6-minute code walkthrough

1. **Repository (45 s)** — Show README, attribution, scope and figures.
2. **Model (60 s)** — Point out encoder/decoder components and four prediction maps.
3. **Training (75 s)** — Explain structure loss, three scales, 15 combinations, optimizer, checkpoints and stopping decision.
4. **Evaluation (60 s)** — Explain metric aggregation and dataset differences.
5. **Demo (120 s)** — Load checkpoint, choose image, run inference, inspect masks and metric outputs.

## Common questions

**Why epoch 19?** The saved checkpoint selection rule maximized image-count-weighted Dice across four validation datasets. Later completed epochs through 90 did not exceed its score.

**Why stop at 90?** Project deadline and diminishing validation improvement. This is a practical choice, not proof that later epochs cannot improve.

**What is new?** Reproducibility engineering and interactive demonstration; robustness experiments are a separately evaluated proposed research extension.

**Dice vs IoU?** Dice = 2TP/(2TP+FP+FN); IoU = TP/(TP+FP+FN). Both measure mask overlap; Dice weights overlap twice.

**Can a random uploaded image have metrics?** Only if ground truth is supplied; otherwise prediction and overlay only.
