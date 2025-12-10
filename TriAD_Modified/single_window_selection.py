# This script outlines the method for selecting the most suspicious window after tri-domain detection by comparing it with training data. The single window will undergo padding for the final step: discord discovery (Merlin).

import numpy as np
import pandas as pd
import time, yaml, argparse

from utils.utils import find_period, pkl_load, sliding_window, pkl_save
from preprocess_data import load_anomaly_smartgrid
from configs.grid_settings import ID, LABEL


def l2norm_rows(x):
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def cos_max_per_row_chunked(X, Y, chunk=4096):
    Nt = X.shape[0]
    out_max = np.full(Nt, -1.0, dtype=np.float32)
    for j in range(0, Y.shape[0], chunk):
        Yj = Y[j:j+chunk]
        for i in range(0, Nt, chunk):
            Xi = X[i:i+chunk]
            blk = np.abs(Xi @ Yj.T)
            blk_max = blk.max(axis=1)
            np.maximum(out_max[i:i+chunk], blk_max, out=out_max[i:i+chunk])
    return out_max

# Refactored cosine-similarity with chunking & precomputed reference slices
def Cos_sim_refac(target_win, period, refer_slices_norm, chunk=4096):
    target_slices = sliding_window(target_win, period, stride=1)
    X = np.asarray(target_slices, dtype=np.float32)
    X = l2norm_rows(X)
    max_cos = cos_max_per_row_chunked(X, refer_slices_norm, chunk=chunk)
    return float(max_cos.min())

def merlin_hit(row, gt_points):
    mw = row['merlin_suspects']
    if isinstance(mw, np.ndarray):
        mw = mw.tolist()
    mw_set = set(map(int, mw))
    return int(len(gt_points.intersection(mw_set)) > 0)

# Get 1D series for both train and test (use channel 0 if data is [T, C])
def to_1d(arr):
    return arr[:, 0] if getattr(arr, "ndim", 1) == 2 else arr


def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f)

    results_dir = "./results"
    output_dir = "./merlin_res"
    id = ID

    dataset_kwargs = CFG["dataset_kwargs"]
    label_cfg = dataset_kwargs["labels"]

    print("\n" + "=" * 80)
    print(f"SINGLE-WINDOW RUN: combine='{LABEL}' | features={dataset_kwargs['feature_columns']}")
    print("=" * 80)

    # Load data
    train_x, valid_x, test_x, test_y = load_anomaly_smartgrid(
        dataset_kwargs["csv_path"],
        key=dataset_kwargs["key"],
        label=label_cfg["columns"],
        combine=label_cfg["combine"],
        feature_cols=dataset_kwargs["feature_columns"],
    )
    
    train_data   = train_x[id]
    test_labels  = test_y[id]
    test_series  = to_1d(test_x[id]).astype(np.float32)
    train_series = np.concatenate(
        [to_1d(train_x[id]), to_1d(valid_x[id])]
    ).astype(np.float32)

    # Period for this dataset
    series_for_period = train_data[:, 0] if getattr(train_data, "ndim", 1) == 2 else train_data
    period_len = find_period(series_for_period, id)

    # Metadata for this dataset
    data = {
        "id": id,
        "period": period_len,
        "anomaly_len": int(np.sum(test_labels == 1)),
        "labels": test_labels,
        "gt_loc": np.where(test_labels == 1)[0],
    }
    all_data = pd.DataFrame([data])

    # --- Tri-window result ---
    tri_fn = f"{results_dir}/tri_res.pt"
    res_notebook = pkl_load(tri_fn)

    if isinstance(res_notebook, dict):
        res_df = pd.DataFrame([res_notebook])
    else:
        res_df = pd.DataFrame(res_notebook)

    # Merge and compute tri-window accuracy
    merged = all_data.merge(res_df, on='id', how='inner')
    tri_acc = float(np.mean(merged['tri_detected'].astype(bool)))

    print(f"Dataset: {id}")
    print(f"Period length: {period_len}")
    print(f"Anomaly windows: {merged['anomaly_len'].iloc[0]}")
    print(f"Anomaly {'DETECTED' if tri_acc == 1.0 else 'MISS'}")
    print(f"Tri-window prediction accuracy: {tri_acc:.3f}")

    # --- Merlin + single-window selection ---
    period_padding = 2
    t = time.time()

    windows = res_df['suspects'].iloc[0]   # list of (start, end) tuples
    period = int(find_period(
        train_x[id][:, 0] if getattr(train_x[id], "ndim", 1) == 2 else train_x[id], id
    ))
    gt = np.where(test_y[id] == 1)[0]

    # Optional drop_10, matching training
    train_series = train_series[len(train_series) // 10:]
    test_len = len(test_series)

    t0 = time.time()
    pad = int(period_padding * period)

    slices = [
        np.arange(max(0, int(s) - pad), min(int(e) + pad, test_len), dtype=int)
        for (s, e) in windows
        if min(int(e) + pad, test_len) > max(0, int(s) - pad)
    ]
    if slices:
        m = min(map(len, slices))
        windows_updated = [a[:m] for a in slices]
        cand_win = np.stack(windows_updated)  # [K, W]
    else:
        windows_updated, cand_win = [], np.array([], dtype=int)

    # Build reference slices
    refer_slices = sliding_window(train_series, period, stride=1)
    refer_slices_norm = l2norm_rows(np.asarray(refer_slices, dtype=np.float32))

    # Inner loop: pick best candidate window by cosine sim
    min_sim = float('inf')
    pred_win = 0

    for win_i, idxes in enumerate(cand_win):
        target_win = test_series[idxes].astype(np.float32)
        sim = Cos_sim_refac(target_win, period, refer_slices_norm)
        if sim < min_sim:
            min_sim = sim
            pred_win = win_i

    elapsed = time.time() - t
    pad_mult = 3 if period <= 100 else 2

    if len(cand_win) > 0:
        base_win = windows_updated[pred_win]
        start = max(0, int(base_win[0]) - int(pad_mult * period))
        end   = min(int(base_win[-1]) + int(pad_mult * period), test_len)
        merlin_win = np.arange(start, end, dtype=int)
    else:
        merlin_win = np.array([], dtype=int)

    # Ground-truth candidate indices
    gt_win = [i for i, idx in enumerate(windows_updated) if np.any(np.isin(idx, gt))] if len(cand_win) > 0 else []

    # Update res_df with single-window + Merlin info
    res_df.loc[:, 'single_win']      = int(pred_win)
    res_df.loc[:, 'filter_time']     = float(elapsed)
    res_df.loc[:, 'merlin_len']      = int(len(merlin_win))
    res_df.loc[:, 'merlin_suspects'] = [np.asarray(merlin_win, dtype=int).tolist()]
    res_df.loc[:, 'gt_win_idx']      = [list(gt_win)]

    gt_points = set(map(int, gt))
    res_df.loc[:, 'merlin_detected'] = res_df.apply(merlin_hit, axis=1, gt_points=gt_points)
    merlin_acc = float(res_df['merlin_detected'].astype(bool).mean())
    print(f"MERLIN(window) detection accuracy: {merlin_acc:.3f}")

    single_preds = sum(res_df.apply(lambda row: row['single_win'] in row['gt_win_idx'], axis=1))
    single_acc = single_preds / len(res_df)
    print(f"Single window detection accuracy: {single_acc:.3f}")

    # Save per-run results
    out_path = f"{output_dir}/{dataset_kwargs['feature_columns']}_merlin_win.pt"
    pkl_save(out_path, res_df)
    print(f"Saved Merlin window results to {out_path}")

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
