# Self-Supervised Anomaly Detection for Harmonics-Rich Smart-Grid Time Series  
**CARLA vs. TriAD: A Design-Centric Comparative Study**

> **NOTE**  
> Model-specific setup and execution instructions are provided in the `README.md` file within each model directory (`CARLA_Modified/`, `TriAD_Modified/`).

## Overview & Methodology

This repository studies how **feature composition, temporal windowing, and label formulation** influence the behaviour of **self-supervised learning (SSL)** models for anomaly detection in **harmonics-rich smart-grid time-series data**.

Two contrastive SSL frameworks are evaluated under a unified experimental design:

- **CARLA** — a feature-agnostic baseline driven primarily by temporal structure and neighborhood consistency in embedding space.
- **TriAD** — a feature-sensitive, multi-domain SSL model that detects anomalies through inconsistencies across temporal, frequency, and residual representations.

Rather than proposing a new detector, this work focuses on **methodological analysis**: identifying *when* SSL models are invariant to feature semantics and *when* representation and feature design materially affect anomaly detection performance.

---

## CARLA: Design and Experimental Role

CARLA is implemented as a **contrastive SSL baseline** to assess feature sensitivity in heterogeneous smart-grid data. Since CARLA has **no built-in feature selection mechanism**, it provides a controlled reference for isolating the influence of temporal structure.

Key design adaptations include:

- **Dynamic feature subset support**, relaxing the original fixed-dimensional input assumption.
- **Training-only z-score normalization** on normal samples.
- **Strict chronological splitting** to prevent temporal leakage.

CARLA operates on **fixed-length sliding windows**, each assigned a **single binary anomaly label** derived from underlying point-level annotations. Multiple window-level labeling modes (e.g., count-based, fraction-based, baseline) are used to test robustness under varying anomaly distributions.

CARLA’s role in this repository is **analytical**, not final optimisation.

---

## TriAD: Feature Engineering and Representation Design

TriAD is evaluated under both **univariate** and **multivariate** formulations to examine the role of feature interaction in anomaly detection.

For each window, TriAD constructs three complementary views:

- **Temporal** — raw time-domain signal  
- **Frequency** — FFT-based harmonic representation  
- **Residual** — deviation from detrended and de-seasonalised baseline  

Augmentations are applied **consistently across channels**, preserving inter-feature relationships. This design enables TriAD to detect anomalies arising from **cross-channel or cross-domain inconsistency**, which are common in industrial systems.

---

## Feature Selection via SFFS (TriAD Only)

Because TriAD is sensitive to feature composition, **Sequential Floating Forward Selection (SFFS)** is used to identify compact and informative feature subsets:

1. Start from an empty feature set  
2. Iteratively add features that improve validation performance  
3. Conditionally remove redundant features  
4. Repeat until convergence  

Selected features define the **input channels to TriAD**. Feature selection is **model-specific** and optimised using F1-score, reflecting the trade-off between missed anomalies and false alarms.

---

## Pointwise Anomaly Scoring (TriAD)

TriAD extends beyond window-level detection through a **multi-stage pointwise scoring pipeline**:

1. Identify domain-specific suspect windows (temporal, frequency, residual)
2. Select a **deep inspection window** with strongest anomaly evidence
3. Detect anomalous subsequences using a discord discovery algorithm
4. Produce final anomaly location and score for evaluation

This design supports both **detection and localisation**, rather than coarse window-level flags.

---

## Experimental Design Summary

- **Dataset:** Multivariate smart-grid time series (50,000 records, 15-min resolution)
- **Anomaly types:** Overload, Transformer fault
- **Label modes:** ANY, BOTH, and individual fault modes
- **Windowing:**  
  - CARLA — fixed-length sliding windows  
  - TriAD — cycle-based temporal windows  
- **Metrics:**  
  - Recall (primary, high miss cost)  
  - F1-score (balanced performance)  
  - Precision (monitored false alarms)  
  - AUC (pointwise scoring support)  

Accuracy is avoided due to severe class imbalance.

---

## Key Observations

- CARLA exhibits **strong feature invariance** across feature subsets and label modes.
- TriAD shows **high sensitivity to feature composition**, particularly under multivariate inputs.
- Longer temporal windows improve detection of gradual anomalies.
- Label formulation significantly affects detectability; stricter definitions (e.g., BOTH) are more reliable.

---

## Limitations

- Dataset diversity is limited.
- Fully unlabeled industrial settings remain inherently ambiguous.
- Metric-driven feature selection may penalise high-recall configurations.
