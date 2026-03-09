# Self-Supervised Anomaly Detection for Harmonics-Rich Smart-Grid Time Series: A Design-Centric Comparison of CARLA and TriAD

> **NOTE**  
> Model-specific setup and execution instructions are provided in the `README.md` file within each model directory (`CARLA_Modified/`, `TriAD_Modified/`).

## Overview & Methodology

This repository studies how **feature composition, temporal windowing, and label formulation** influence the behaviour of **self-supervised learning (SSL)** models for anomaly detection in **harmonics-rich smart-grid time-series data**.

Two contrastive SSL frameworks are evaluated under a unified experimental design:

- **CARLA** — a feature-agnostic baseline driven primarily by temporal structure and neighborhood consistency in embedding space.
- **TriAD** — a feature-sensitive, multi-domain SSL model that detects anomalies through inconsistencies across temporal, frequency, and residual representations.

Rather than proposing a new detector, this work focuses on **methodological analysis** identifying: 
- *When* SSL models are invariant to feature semantics.
- *When* representation and feature design materially affect anomaly detection performance.

---

## CARLA: Design and Experimental Role

CARLA is implemented as a **contrastive SSL baseline** to assess feature sensitivity in heterogeneous smart-grid data. Since CARLA has **no built-in feature selection mechanism**, it provides a controlled reference for isolating the influence of temporal structure.

Key design adaptations include:

- Support for a **variable** number of input features, removing the fixed-dimensional input constraint.
- Apply **z-score normalization** on training-only samples.
- **Strict chronological splitting** prevents temporal leakage.

Evaluation is based on **multiple window-level labeling modes (e.g., count-based, fraction-based, baseline)**, enabling assessment under varying anomaly distributions.

CARLA’s role in this repository is **analytical**, not final optimisation.

---

## TriAD: Feature Engineering and Representation Design

TriAD is implemented and analysed in **multivariate formulation** with the original **univariate TriAD** is retained as a **comparative benchmark** to isolate the effect of feature interaction.

For each window, the multivariate TriAD constructs three complementary views in the **temporal**, **frequency** and **residual** domains. Augmentations are applied **consistently across channels**, preserving inter-feature relationships. This design enables TriAD to detect anomalies arising from **cross-channel and cross-domain inconsistencies**, which are characteristic of industrial fault scenarios and cannot be captured by univariate formulations alone.

To control redundancy and evaluate feature relevance, **Sequential Floating Forward Selection (SFFS)** is applied to identify compact feature subsets for TriAD experiments. SFFS is used only as a **selection mechanism** to determine which features are provided as input channels to TriAD optimised using **F1-score**, reflecting the **trade-off between missed anomalies and false alarms**.

In the original TriAD formulation, pointwise anomaly scoring is selection-based, where the most suspicious window or subsequence is chosen to represent the final anomaly decision. In this work, the scoring strategy is modified to an **additive-based** formulation, which aggregates anomaly evidence from multiple sources, including **domain-specific windows (temporal, frequency, residual)**, the **most suspicious single window**, and an **extended suspicious region identified via MERLIN**.

---

## Experimental Design

- **Dataset:** Multivariate smart-grid time series (50,000 records, 15-min resolution)
- **Anomaly types:** Overload Conditions, Transformer Faults
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

- Dataset records are not enough for concise evaluation.
- Fully unlabeled industrial settings remain inherently ambiguous.
- Metric-driven (e.g. F1 score) feature selection may penalise high-recall configurations.
