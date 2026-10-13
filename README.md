# SGCC Electricity Theft Detection

## Problem
Binary classification of electricity theft from 42K users × 1,034 days of smart meter data. 8.5% positive class.

## Approach
End-to-end pipeline: cleaning → EDA → feature engineering → LightGBM/XGBoost.

## Key results (v1 final)
- F1: 0.492
- Precision: 0.543
- Recall: 0.450
- AUC-PR: 0.496

- ## Results (v1 final — LightGBM)

### Overall metrics
| Metric | Value |
|---|---|
| Accuracy | 0.920 |
| AUC-ROC | 0.858 |
| AUC-PR (theft class) | 0.496 |

### Per-class performance
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.95 | 0.96 | 0.95 | 7,752 |
| Theft | 0.54 | 0.45 | 0.49 | 723 |
| **Macro avg** | 0.75 | 0.71 | 0.72 | 8,475 |
| **Weighted avg** | 0.91 | 0.91 | 0.91 | 8,475 |

### Confusion matrix
|  | Predicted Normal | Predicted Theft |
|---|---|---|
| **Actual Normal** | 7,478 (TN) | 274 (FP) |
| **Actual Theft** | 398 (FN) | 325 (TP) |

### What the numbers mean
- **Catch rate:** 45% of theft users flagged (325/723)
- **Field utility:** 54% of flagged users are real thieves (325/599)
- **False alarm rate:** 3.5% of honest users incorrectly flagged (274/7752)
- **AUC-PR:** 0.496 — 5.8× above random baseline (0.085)

### Version comparison
| Version | Features | F1 (theft) | Precision | Recall |
|---|---|---|---|---|
| v1 (final) | 40 | **0.492** | **0.543** | 0.450 |
| v2 | 46 | 0.501 | 0.506 | 0.497 |
| v4 | 40 | 0.489 | 0.464 | 0.516 |
| v5 | 37 | 0.480 | 0.507 | 0.456 |
| v6 | 43 | 0.480 | 0.420 | 0.559 |

**v1 is chosen as final:** highest precision, most balanced profile, simplest feature set.

## What worked
- NaN structure features (2016 streaks, peak-month ratios)
- Usage level + volatility
- Threshold tuning, scale_pos_weight

## What didn't work (tested and rejected)
- Autocorrelation features (acf_lag1/7/30)
- Flatness / constant-streak features
- Festival response ratios
- State-shift counts
- Year-over-year deltas
- Within-peak volatility

## Honest notes
Many public SGCC notebooks report F1 > 0.90. These leak by balancing before splitting. This pipeline splits first — F1 ≈ 0.49 is the honest tabular ceiling.

## Reproduce
Developed on Kaggle. Upload the SGCC dataset, adjust paths at the top of the notebook.
