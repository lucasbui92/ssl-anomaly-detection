# Adapted from: TriAD (https://github.com/pseudo-Skye/TriAD).
# Modifications by Thuan Anh Bui, 2025.
# Changes: adjusted evaluation techniques and modified graph visualization

import sys, os, yaml, argparse, csv
sys.path.insert(0, '../')

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from itertools import groupby
from operator import itemgetter

from utils.utils import pkl_load 
from utils.eval_metrics import trad_metrics, aff_metrics, f1_prec_recall_K, f1_prec_recall_PA
from preprocess_data import load_anomaly_smartgrid
from configs.grid_settings import ID, LABEL, GLOBAL_METRICS_FILE
from pathlib import Path

def to_index_array(win, n_total):
    """
    Normalize a window to integer indices in [0, n_total).
    Supports:
      - tuple/list/ndarray of (start, end) inclusive
      - iterable of indices
    """
    if win is None:
        return np.array([], dtype=int)

    # Treat (start,end) when it's tuple/list OR a 1D ndarray of length 2
    if (
        (isinstance(win, (tuple, list)) and len(win) == 2 and all(isinstance(k, (int, np.integer)) for k in win))
        or (isinstance(win, np.ndarray) and win.ndim == 1 and win.size == 2 and np.issubdtype(win.dtype, np.integer))
    ):
        s, e = int(win[0]), int(win[1])
        s = max(0, s); e = min(n_total - 1, e)
        if e < s:
            return np.array([], dtype=int)
        return np.arange(s, e + 1, dtype=int)

    # Otherwise assume it's a collection of indices
    arr = np.asarray(win, dtype=int).ravel()
    arr = arr[(arr >= 0) & (arr < n_total)]
    return np.unique(arr)

def overlap_frac(a_idx, b_idx):
    """Jaccard-like overlap: |A ∩ B| / |A ∪ B| (0 if both empty)."""
    if len(a_idx) == 0 and len(b_idx) == 0:
        return 0.0
    A = set(map(int, a_idx)); B = set(map(int, b_idx))
    inter = len(A & B)
    uni = len(A | B)
    return (inter / uni) if uni > 0 else 0.0


def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f)

    id = ID
    results_dir = Path("./merlin_res")

    dataset_kwargs = CFG["dataset_kwargs"]
    label_cfg = dataset_kwargs["labels"]

    print("\n" + "=" * 80)
    print(f"EVALUATE RUN: combine='{LABEL}' | features='{dataset_kwargs['feature_columns']}'")
    print("=" * 80)

    # Load data
    train_x, valid_x, test_x, test_y = load_anomaly_smartgrid(
        dataset_kwargs["csv_path"],
        key=dataset_kwargs["key"],
        label=label_cfg["columns"],
        combine=label_cfg["combine"],
        feature_cols=dataset_kwargs['feature_columns'],
    )

    print("Dataset: Smart Grid Monitoring")
    print(f"ID: {id}")

    test_data = test_x[id]
    gt_label  = test_y[id]

    # Load Merlin results for this (combine, feature) run
    merlin_file = results_dir / f"{dataset_kwargs['feature_columns']}_merlin_win.pt"
    merlin_win = pkl_load(str(merlin_file))

    # Ensure DataFrame-like indexing
    if isinstance(merlin_win, dict):
        merlin_df = pd.DataFrame([merlin_win])
    else:
        merlin_df = pd.DataFrame(merlin_win)

    row = merlin_df[merlin_df["id"] == id]

    tri_wins   = row['suspects'].values[0]
    m_win      = row['merlin_suspects'].values[0]
    single_idx = row['single_win'].values[0]
    deep_win   = np.asarray(tri_wins[single_idx], dtype=int)

    N = len(test_data)
    # Normalize inputs to index arrays (robust to tuple vs index-list)
    deep_idx   = to_index_array(deep_win, N)
    merlin_idx = to_index_array(m_win, N)

    # TriAD suspects: normalize each candidate window
    cand_windows = [to_index_array(w, N) for w in tri_wins]

    # Build pointwise votes
    tri_votes = np.zeros(N, dtype=float)
    for w in cand_windows:
        if len(w):
            tri_votes[w] += 1.0     # each candidate window votes for its points

    merlin_mask = np.zeros(N, dtype=float)
    if len(merlin_idx):
        merlin_mask[merlin_idx] = 1.0

    deep_mask = np.zeros(N, dtype=float)
    if len(deep_idx):
        deep_mask[deep_idx] = 1.0

    # TODO: Adjust weight sources if wanted
    TRI_W, MERLIN_W, DEEP_W = 1.0, 1.0, 1.0
    pw_score = TRI_W * tri_votes + MERLIN_W * merlin_mask + DEEP_W * deep_mask

    # Threshold = MEDIAN of non-zero scores
    nonzero = pw_score[pw_score > 0]
    threshold = float(np.median(nonzero)) if nonzero.size else 0.0

    # Predictions
    pred_label = np.zeros(N, dtype=int)
    pred_label[pw_score >= threshold] = 1

    # "Window magic correction": if MERLIN had no hits inside deep window, force deep window to 1s
    # if len(deep_idx) and merlin_mask[deep_idx].max() == 0:
    #     print('window magic correction !!')
    #     pred_label[deep_idx] = 1

    # Compute the metrics
    acc, prec, recall, f1 = trad_metrics(gt_label, pred_label)
    f1_pa, prec_pa, recall_pa = f1_prec_recall_PA(pred_label, gt_label)
    f1_pak_auc, prec_pak_auc, recall_pak_auc = f1_prec_recall_K(pred_label, gt_label)
    prec, recall = aff_metrics(pred_label, gt_label)

    # Print summary
    print("Traditional Metrics:")
    print(f"  F1 Score: {f1:.4f}\n")

    print("PA:")
    print(f"  F1 Score: {f1_pa:.4f}\n")

    print("PA%K - AUC:")
    print(f"  Precision: {prec_pak_auc:.4f}")
    print(f"  Recall: {recall_pak_auc:.4f}")
    print(f"  F1 Score: {f1_pak_auc:.4f}\n")

    print("Affinity:")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall: {recall:.4f}")

    with open(Path(GLOBAL_METRICS_FILE), "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        # Encode as "Voltage (V)|Current (A)|Power Consumption (kW)"
        features_str = "|".join(dataset_kwargs["feature_columns"])

        writer.writerow([
            label_cfg["combine"],
            features_str,
            f"{acc:.4f}",
            f"{f1:.4f}",
            f"{f1_pa:.4f}",
            f"{prec_pak_auc:.4f}",
            f"{recall_pak_auc:.4f}",
            f"{f1_pak_auc:.4f}",
            f"{prec:.4f}",
            f"{recall:.4f}",
        ])
    print(f"Appended metrics to {GLOBAL_METRICS_FILE}")

    # Ground-truth anomaly indices
    gt = np.where(gt_label == 1)[0]

    signal_1d = np.asarray(test_data)
    if signal_1d.ndim == 2:
        signal_1d = signal_1d[:, 0]
    elif signal_1d.ndim > 2:
        signal_1d = signal_1d.reshape(signal_1d.shape[0], -1)[:, 0]

    N = signal_1d.shape[0]
    x = np.arange(N)

    smooth = pd.Series(signal_1d).rolling(window=25, center=True).mean().to_numpy()
    smooth = np.where(np.isnan(smooth), signal_1d, smooth)

    gt_segments = []
    for _, g in groupby(enumerate(gt), lambda ix: ix[0] - ix[1]):
        seg = list(map(itemgetter(1), g))
        gt_segments.append(seg)

    demo_gt = gt
    if gt_segments and len(deep_idx) > 0:
        center = (deep_idx[0] + deep_idx[-1]) / 2.0

        def seg_distance(seg):
            return abs(center - (seg[0] + seg[-1]) / 2.0)

        demo_gt = min(gt_segments, key=seg_distance)

    # --------------------------------------------------
    # Choose zoom window (prefer deep window, else GT)
    # --------------------------------------------------
    zoom_left, zoom_right = 0, N - 1
    if len(deep_idx) > 0:
        center = (deep_idx[0] + deep_idx[-1]) // 2
        half_win = 200
        zoom_left = max(center - half_win, 0)
        zoom_right = min(center + half_win, N - 1)
    elif len(demo_gt) > 0:
        center = (demo_gt[0] + demo_gt[-1]) // 2
        half_win = 200
        zoom_left = max(center - half_win, 0)
        zoom_right = min(center + half_win, N - 1)

    eval_dir = Path("eval_demo")
    eval_dir.mkdir(parents=True, exist_ok=True)

    existing_ids = []
    for p in eval_dir.glob("*_triad_merlin_windows.png"):
        try:
            stem = p.stem  # e.g. "3_triad_merlin_windows"
            first_part = stem.split("_", 1)[0]
            existing_ids.append(int(first_part))
        except ValueError:
            pass

    next_id = (max(existing_ids) + 1) if existing_ids else 1

    # ------------------------------ Visualize Merlin + TriAD windows ------------------------------
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax1, ax2 = ax

    # Top: smoothed signal + demo ground truth
    ax1.plot(x, smooth, color='steelblue', alpha=0.5, linewidth=1.0)
    if len(demo_gt) > 0:
        ax1.plot(demo_gt, smooth[demo_gt], color='red', linewidth=2.0)
    ax1.set_ylabel('Amplitude', size=12)
    ax1.set_title("Ground truth", size=12)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)

    # Bottom: only GT segment, TriAD deep window, Merlin window
    ax2.set_xlabel('Timestamps', size=12)
    ax2.set_ylabel('Anomaly windows', size=12)
    ax2.set_yticks([])
    ax2.tick_params(axis='x', labelsize=12)

    handles, labels = [], []

    # Ground truth window
    if len(demo_gt) > 0:
        gt_patch = ax2.axvspan(demo_gt[0], demo_gt[-1], color='yellow', alpha=0.4)
        handles.append(gt_patch)
        labels.append('Ground truth')

    # TriAD deep window
    if len(deep_idx) > 0:
        triad_patch = ax2.axvspan(deep_idx[0], deep_idx[-1], color='0.7', alpha=0.7)
        handles.append(triad_patch)
        labels.append('TriAD deep window')

    # Merlin window
    if len(merlin_idx) > 0:
        merlin_patch = ax2.axvspan(merlin_idx[0], merlin_idx[-1], color='orange', alpha=0.6)
        handles.append(merlin_patch)
        labels.append('Merlin candidate window')

    ax1.set_xlim(zoom_left, zoom_right)
    ax2.set_xlim(zoom_left, zoom_right)

    if handles:
        ax2.legend(handles, labels, fontsize=12, framealpha=0.9, loc='upper left')

    plt.subplots_adjust(hspace=0.1)

    fig.savefig(
        eval_dir / f"{next_id}_triad_merlin_windows.png",
        format='png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    # ------------------------------ Visualize point-wise detection results ------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    ax1, ax2 = ax

    # Top: GT
    ax1.plot(x, smooth, color='steelblue', alpha=0.5, linewidth=1.0)
    if len(demo_gt) > 0:
        ax1.plot(demo_gt, smooth[demo_gt], color='red', linewidth=2.0)
    ax1.set_yticks([])
    ax1.set_title("Ground truth", size=12)

    # Bottom: prediction (orange segments)
    pred_idx = np.where(pred_label == 1)[0]
    indices = []
    for k, g in groupby(enumerate(pred_idx), lambda ix: ix[0] - ix[1]):
        indices.append(list(map(itemgetter(1), g)))

    ax2.plot(x, smooth, color='steelblue', alpha=0.5, linewidth=1.0)
    for idx in indices:
        ax2.plot(x[idx], smooth[idx], color='orange', linewidth=2.0)
    ax2.set_yticks([])
    ax2.set_title("Prediction", size=12)
    ax2.set_xlabel('Timestamps', size=12)
    ax2.tick_params(axis='x', labelsize=12)

    ax1.set_xlim(zoom_left, zoom_right)
    ax2.set_xlim(zoom_left, zoom_right)
    plt.subplots_adjust(hspace=0.5)

    fig.savefig(
        eval_dir / f"{next_id}_pointwise_detection.png",
        format='png',
        dpi=300,
        bbox_inches='tight'
    )
    plt.close(fig)

    print(f"Saved evaluation figures {next_id}_*.png to {eval_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        type=str,
        default=f"./configs/triad_train_{ID}.yml",
        help="Path to YAML config for this single run.",
    )
    args = parser.parse_args()
    main(args.cfg)
