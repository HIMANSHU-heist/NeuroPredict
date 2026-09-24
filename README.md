# NeuroPredict: Does Adding Heart Signals to Brain Signals Help Predict Mental Effort?

An honest, externally tested study of **EEG + ECG fusion** for classifying **rest vs mental task**.
Trained on one public dataset, then tested on a completely different one (different people, lab, task and EEG reference).

> **One-line summary:** On unseen data, an EEG + ECG ensemble beat EEG-only by **+7.7 points** (64.6% vs 56.9%, p = 0.0016), but the heart signal did much of the work and the EEG model transferred poorly.

---

## Research question

Does an EEG + ECG ensemble generalize better than EEG-only when tested on data the model has never seen: different people, a different lab and a different task?

## Datasets

| | EEGMAT (train) | ds003838 (external test) |
|---|---|---|
| Source | PhysioNet | OpenNeuro |
| People | 36 | 65 |
| Task | Mental arithmetic | Digit-span working memory |
| Original reference | Linked ears | FCz |
| Used | 19 standard 10-20 channels, 500 Hz | Same 19 channels, resampled to 500 Hz |

Both datasets are re-referenced to **average reference**. A third dataset (MPD-DF, driving fatigue) was evaluated and **rejected** because its labels do not map onto rest-vs-task.

## Method

- **EEG:** 1-40 Hz band-pass, average reference, 4 s windows (2 s step), per-channel z-score, small custom 1D-CNN (3 conv blocks).
- **ECG:** mean heart rate, SDNN, RMSSD, then logistic regression.
- **Ensemble:** weighted average of both probabilities (weights from training-side accuracy only).
- **Internal test:** leave-one-subject-out (36 folds) x 5 seeds = 180 training runs.
- **External test:** final models trained on all 36 EEGMAT people, tested on 65 unseen people. The notebook automatically checks window length, step, channels, sampling rate and reference, and stops on any mismatch.
- **Statistics:** bootstrap 95% CI (2,000 resamples), Wilcoxon signed-rank, Cohen's d, ROC-AUC, confusion matrices, ablation, nested method selection.

## Key results

**Internal (EEGMAT, 5 seeds, balanced accuracy)**

| EEG-only | ECG-only | Ensemble |
|---|---|---|
| 70.6% | 68.1% | **76.1%** (wins in 5/5 seeds) |

**External (ds003838, 65 new people)**

| Model | Balanced acc. | 95% CI | AUC |
|---|---|---|---|
| EEG-only | 56.9% | 53.1-61.5 | **0.806** |
| ECG-only | 60.0% | 55.4-65.4 | 0.675 |
| Ensemble | **64.6%** | 59.2-70.0 | 0.763 |
| Stacked meta-classifier | 70.0% | n/a | n/a |

- Ensemble vs EEG-only: mean **+7.7 points**, 95% CI +3.8 to +12.3, Wilcoxon p = 0.0016, Cohen's d = 0.423 (small-to-medium). 10 wins, 55 ties, 0 losses.
- The gain comes from fixing the EEG model's bias toward "task": rest correctly identified went from 9/65 to 19/65.

## What is and is not claimed

**Claimed**
- At the 0.5 cut-off, the ensemble beats EEG-only on new people.
- It beats EEG-only in all 5 internal seeds.

**Not claimed**
- Better ranking: EEG-only has the higher AUC (gap not confirmed, CI -0.095 to +0.007).
- Beating ECG-only: not confirmed on the external set (CI includes 0).
- Readiness for real-world use: 64.6% is far too low for safety applications.

## Honesty log

The report includes a full bug and correction table: no fixed seed, 2 s vs 4 s window mismatch, EEG reference mismatch, channel-count bug, outdated ensemble weights (re-run still open), a statistical test that could not fail (dropped), and over-strong wording on effect size and AUC.

## Limitations

- Each person gives only 2 test points (scores are 0, 50 or 100%), so CIs are wide.
- Only one successful external dataset.
- ECG features are noisy (SDNN/RMSSD), so the ECG model is effectively a heart-rate model.
- Only 19 of 63 channels used on the external set.
- Internal seeds share the same 36 people, so they are not independent tests.
- Nested selection check covers only 2 of 5 combination schemes.

## Repository structure

```
.
├── README.md
├── report/
│   └── NeuroPredict_Final_Report.pdf
└── notebooks/
    ├── 1_weighted_ensemble_v3.ipynb        # Colab: LOSO x 5 seeds, trains final EEG/ECG models
    ├── 2_external_validation_ds003838.ipynb # Kaggle: external test, stats, ablation
    └── 3_eeg_ecg_eda_eegmat.ipynb          # Colab: band power, HRV, t-tests, correlation
```

## How to reproduce

1. Run **notebook 3** to reproduce the EDA (band power and HRV, rest vs task).
2. Run **notebook 1** to train and save the final EEG model, ECG model and settings file (long runs are checkpointed and resume-safe).
3. Run **notebook 2** to load the saved models and evaluate on ds003838 (settings are verified automatically).

Main libraries: PyTorch, scikit-learn, MNE, NeuroKit2, NumPy, SciPy, pandas.

## Future work

- Re-run the external test with corrected ensemble weights (~0.509 / 0.491).
- Clean ECG beat detection; add robust HRV features.
- Try real EEGNet and band-power classifiers; domain adaptation; more channels.
- Second external dataset with a true rest-vs-task design.
- Wearable-friendly version (few forehead channels + wrist pulse sensor) and per-user calibration.

## Data sources

- EEGMAT: PhysioNet
- ds003838: OpenNeuro

Please follow each dataset's own license and citation requirements.

## License

Add your preferred license here (e.g. MIT).

## Author
Himanshu Bendale
Second-year undergraduate student. Feedback and issues welcome.
