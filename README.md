# SGCC Electricity Theft Detection

Binary classification of electricity theft from smart meter time series using gradient boosting.

## Problem

**Dataset:** State Grid Corporation of China (SGCC) smart meter readings.

- **42,372 users** × **1,034 days** of daily electricity consumption (Jan 2014 – Oct 2016)
- **Binary target:** `FLAG` — 0 = normal, 1 = confirmed theft
- **Class imbalance:** 8.5% theft (3,615 positive / 38,757 negative)
- **No metadata** — only raw consumption values, no region, income, or meter type
- **25.6% missing values** — gaps in meter reporting

The goal is to identify likely theft users from consumption patterns alone, so inspectors can prioritize high-risk customers.

## Approach

End-to-end ML pipeline:

1. **Cleaning** — date parsing, chronological sort, NaN preserved (no user dropping)
2. **Exploratory Data Analysis** — 10+ hypotheses tested across NaN, usage, shape, seasonal, and calendar dimensions
3. **Feature Engineering** — 40 tabular features per user
4. **Modeling** — LightGBM + XGBoost with `scale_pos_weight` and validation-tuned decision threshold
5. **Evaluation** — split-first, no leakage, honest metrics on theft class

### Critical methodology choices

- **Split before any preprocessing** — train/test split happens at the top, no information leaks
- **No synthetic oversampling** — used `scale_pos_weight` instead of SMOTE to avoid leakage
- **Threshold tuned on validation**, not fixed at 0.5 — optimized for F1 on theft class
- **Full test-set report** — both classes' precision, recall, F1 shown

## Key Findings

### What works

| Signal | Lift vs base |
|---|---|
| NaN streak in 2016 peak months | **6.0×** |
| NaN ratio during 2016 summer (>80% missing) | **5.8×** |
| Longest NaN streak in 2016 (>180 days) | **5.2×** |
| NaN on high-usage days in 2016 | **4.3×** |
| Top usage decile | **2.2×** |
| Volatility (`cv`, `max_min_ratio`) | Moderate |

**Dominant signal:** NaN structure — particularly **2016 gaps**, when meter coverage was 98%. Missing readings in 2016 are abnormal; in 2014–2015, they were common (60–70% coverage).

### What didn't work (tested and rejected)

| Feature family | Result |
|---|---|
| Autocorrelation (acf_lag1, 7, 30) | d ≈ 0.2–0.34 univariate, but **hurt precision** when added to model |
| Flatness / constant-streak counts | Reverse signal — thieves *less* flat than honest |
| Festival response ratios | Flat — no separation across quantiles |
| State-shift counts | Weak, non-monotonic |
| Year-over-year deltas | Marginal; peak-season deltas gave +0.01 but unstable |
| Within-peak volatility | No signal |
| Shape features (spike counts, drawdown) | No clean monotonic lift |

**Key negative insight:** Univariate EDA signals don't always help in the multivariate model. acf features dominated feature importance (#1, #2, #7) yet hurt precision by ~12 points.

## Results (v1 — final model)

**Model:** LightGBM, 40 features, threshold = 0.67 (tuned on validation)

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
| **Theft** | **0.54** | **0.45** | **0.49** | **723** |
| Macro avg | 0.75 | 0.71 | 0.72 | 8,475 |
| Weighted avg | 0.91 | 0.91 | 0.91 | 8,475 |

### Confusion matrix

|  | Predicted Normal | Predicted Theft |
|---|---|---|
| **Actual Normal** | 7,478 (TN) | 274 (FP) |
| **Actual Theft** | 398 (FN) | 325 (TP) |

### Interpretation

- **Catch rate:** 45% of theft users correctly flagged (325 / 723)
- **Field utility:** 54% of flagged users are real thieves (325 / 599)
- **False alarm rate:** 3.5% of honest users incorrectly flagged (274 / 7,752)
- **AUC-PR of 0.496** is 5.8× the random baseline (0.085)

For deployment as a prioritization tool: the model flags 599 users for inspection, ~325 of whom are confirmed thieves. Random inspection of 599 users would yield ~52 thieves — a **6.3× improvement**.

## Version comparison

| Version | Features | F1 (theft) | Precision | Recall |
|---|---|---|---|---|
| **v1 (final)** | 40 | **0.492** | **0.543** | 0.450 |
| v2 | 46 | 0.501 | 0.506 | 0.497 |
| v4 | 40 | 0.489 | 0.464 | 0.516 |
| v5 | 37 | 0.480 | 0.507 | 0.456 |
| v6 | 43 | 0.480 | 0.420 | 0.559 |

**v1 is chosen as final** — highest precision, best balanced profile, simplest feature set. Versions v2, v4, v5, v6 tested additional features (deltas, autocorrelation, shape) that either didn't improve F1 or degraded precision.

## Honest notes

Many public SGCC notebooks report F1 > 0.90. **These results leak.** Common leakage:

1. **Balancing before splitting** — synthetic test samples contaminate evaluation
2. **Scaling before splitting** — test statistics leak into train
3. **Threshold tuning on test** — optimistic F1

This pipeline avoids all three. **F1 ≈ 0.49 is the honest ceiling for tabular features on this dataset.**

Further improvement likely requires:
- **Sequence models** (TCN, CNN) fed the raw 1,034-day series
- **Peer-group features** — "user vs similar users" comparisons (requires region metadata not available)
- **Autoencoder reconstruction error** — captures anomaly without supervision

## Reproduce

Developed on Kaggle. To run:

1. Upload the SGCC dataset to Kaggle (`bensalem14/sgcc-dataset`)
2. Adjust `/kaggle/input/...` paths at the top of the notebook
3. Run all cells in order

### Requirements
