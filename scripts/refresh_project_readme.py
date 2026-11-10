"""Update the team README after merging the latest origin/main.

Usage: python scripts/refresh_project_readme.py
Refuses unknown README layout and avoids touching unrelated sections.
"""
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1]
readme = root / 'README.md'
text = readme.read_text(encoding='utf-8-sig')
if '## 11. Verification and Progress' not in text or '## 12. Experiment Reproducibility' not in text:
    sys.exit('STOP: README layout changed; edit manually instead of risking overwrite')
if '## Verified research reproduction and demonstration' not in text and '## 13. Verified Research Reproduction and Demonstration' not in text:
    sys.exit('STOP: GitHub updated reproduction section not found; merge origin/main first')
replacement = '''## 11. Verification and Progress

### Completed and verified

- Baseline environment and PVTv2-B2 pretrained-encoder compatibility checks on an NVIDIA RTX 4050 Laptop GPU.
- Full official-style three-scale, 15-combination training with resumable optimizer/model state.
- 90 of 199 planned epochs, completed across 18 five-epoch sessions; training deliberately frozen before the coursework deadline.
- Model selection on four validation datasets; best image-count-weighted Dice 0.849587 at epoch 19.
- Evaluation of 602 validation images, with dataset-level Dice/IoU and saved per-image outputs.
- Interactive Streamlit demo supporting dropdown image selection, user uploads, ground-truth matching and visual comparison.
- Automated verification of training session scheduling, evaluation behavior and demo panel functions.
- Reproducible figures generated from recorded CSVs with a figure manifest.

### Limitations and pending work

- The planned 199-epoch training schedule was not completed; no improvement after epoch 19 was recorded through epoch 90.
- The four evaluation datasets were used for checkpoint selection, so these metrics are *validation*, not independent held-out test scores.
- Mokshan's robustness extension must be reported separately with saved clean/degraded results before it is described as complete.
- The live demo depends on locally stored checkpoint and dataset files, which are not in Git history.

The earlier batch-size-4 smoke test did not demonstrate the full protocol by itself. The subsequent full training run did use the intended loss and scales; see `docs/training_progress.md` and `docs/final_demo.md`.

'''
text, count = re.subn(r'## 11\. Verification and Progress\s*.*?(?=## 12\. Experiment Reproducibility)', replacement, text, flags=re.S)
if count != 1: sys.exit('STOP: Could not safely replace progress section')
text = text.replace('## Verified research reproduction and demonstration', '## 13. Verified Research Reproduction and Demonstration')
intro = '''### Current repository layout and entry points

```text
README.md                         Team overview and reproduced results
train_polyp.py                    Training, checkpoint resume and validation
lib/                              Original PVT-GCASCADE architecture
utils/                            Data loading, loss, checkpoint/evaluation helpers
scripts/web_demo_polyp.py         Streamlit dashboard
scripts/demo_polyp.py             Single-image inference/panel
scripts/evaluate_polyp.py         Four-dataset evaluation
scripts/plot_results.py           Figures from recorded CSV/JSON artifacts
scripts/run_polyp_session.ps1     Five-epoch resumable training launcher
tests/                            Automated evaluation and training tests
results/baseline/                 Lightweight 90-epoch and final metric records
docs/images/                      Validation figures and source manifest
docs/final_demo.md                Local demo and evaluation details
docs/training_progress.md         Frozen training progress
```

### Presentation quick start

From the repository root, with the compatible `gcascade` environment active and the local model/data files available:

```powershell
python -m unittest -v tests.test_training_sessions tests.test_polyp_evaluation tests.test_demo_polyp
python scripts/plot_results.py
python -m streamlit run scripts/web_demo_polyp.py
```

For the demo, use the sidebar to select the saved `*-best.pth` file and `data/polyp/TestDataset`, then choose **CVC-ClinicDB** and an image such as **171.png**. The per-image Dice/IoU appear when a matching mask is available. Custom uploads without a mask have a prediction but no ground-truth metrics.

For the panel code map and known limitations, see [Repository and viva guide](docs/REPOSITORY_GUIDE.md).

'''
anchor='## 14. Acknowledgments and Citation'
if anchor not in text: sys.exit('STOP: Acknowledgments heading not found')
if '### Current repository layout and entry points' not in text:
    text=text.replace(anchor,intro+anchor,1)
# Drop stale line that implies full training has not been performed.
text=text.replace('Before full training, we must verify that the complete official training loop runs successfully with the 15-combination loss, all three scales and validation.','The completed baseline run verified the official-style 15-combination loss, three training scales and validation; see Section 11 and the recorded training reports.')
text=text.replace('**Do not treat prior partial smoke tests as successful full-protocol training.**','**Early smoke tests alone did not verify the full training protocol; the subsequent recorded training run did.**')
readme.write_text(text,encoding='utf-8',newline='\n')
print('PASS: README sections refreshed. Review git diff README.md before committing.')
