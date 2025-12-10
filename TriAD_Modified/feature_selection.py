# Created by: Thuan Anh Bui (2025)
# Description: Implements automated feature selection pipeline.

import os, sys, subprocess, yaml, csv

import pandas as pd
from pathlib import Path
from configs.grid_settings import ID, LABEL, TARGET, GLOBAL_METRICS_FILE

BASE_CFG_PATH = f"./configs/triad_train_{ID}.yml"
TMP_CFG_PATH = Path("./configs/fs_tmp.yml")  # single reusable config
CFG_DIR = Path("./configs")
CFG_DIR.mkdir(parents=True, exist_ok=True)
SCORE_CACHE = {}    # Store subsets that have been trained


def load_all_features_from_yaml():
    with open(BASE_CFG_PATH, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    feat_list = base_cfg["dataset_kwargs"]["feature_columns"]
    return list(feat_list)

def make_run_name(combine_mode, feature_subset):
    """Create a short run name from combine mode and features."""
    short_feats = [f.split("(")[0].strip().replace(" ", "") for f in feature_subset]
    return f"{combine_mode}__{'_'.join(short_feats)}"

def write_cfg_for_subset(combine_mode, feature_subset, run_name):
    with open(BASE_CFG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["dataset_kwargs"]["labels"]["combine"] = combine_mode
    cfg["dataset_kwargs"]["feature_columns"] = feature_subset
    cfg["run_name"] = run_name
    cfg["results_dir"] = f"./results/{run_name}"
    cfg["merlin_output_dir"] = f"./merlin_res"

    # Always overwrite the same temp file
    with open(TMP_CFG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return str(TMP_CFG_PATH)

def run_full_pipeline(combine_mode, feature_subset):
    # Reuse score of stored subset, avoid duplications later on
    key = (combine_mode, tuple(sorted(feature_subset)))
    if key in SCORE_CACHE:
        return SCORE_CACHE[key]

    run_name = make_run_name(combine_mode, feature_subset)
    cfg_path = write_cfg_for_subset(combine_mode, feature_subset, run_name)

    python_exe = sys.executable  # this is the Python running feature_selection.py

    try:
        # 1) Train
        subprocess.run([python_exe, "train.py", "--cfg", cfg_path], check=True)

        # 2) Window selection
        subprocess.run([python_exe, "single_window_selection.py", "--cfg", cfg_path], check=True)

        # 3) Evaluate
        subprocess.run([python_exe, "evaluate.py", "--cfg", cfg_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Pipeline failed for subset {feature_subset} (run_name={run_name})")
        print(f"        Command: {e.cmd}")
        print(f"        Return code: {e.returncode}")
        return None

    # 4) Sequential Floating Forward Selection (SFFS) needs to know when to stop
    metrics_file = Path(GLOBAL_METRICS_FILE)
    if not metrics_file.exists():
        print(f"[WARN] Metrics file not found!!")
        return None

    # Read only the last data row via DictReader
    last_row = None
    with open(metrics_file, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            last_row = row

    if last_row is None:
        print(f"[WARN] Metrics file has no data rows!")
        return None

    try:
        f1_score = float(last_row["F1_PA"])
        SCORE_CACHE[key] = f1_score
    except (KeyError, ValueError) as e:
        print(f"[ERROR] Failed to parse F1 Score from row: {last_row}")
        return None

    print(f"[PIPELINE] {run_name} -> F1 Score={f1_score:.4f}")

    # CLEANUP SECTION (removes .pt and .pkl files after each run)
    # Delete MODEL checkpoint(s)
    for f in Path("./results").glob("*.pt"):
        f.unlink()

    # Delete TRAINED .pkl artifacts
    for f in Path("./trained").glob("*.pkl"):
        f.unlink()

    # Delete MERLIN window output
    for f in Path("./merlin_res").glob("*.pt"):
        f.unlink()

    return f1_score

def sffs_selection(combine_mode, target_size=13):
    """
    Sequential Floating Forward Selection (SFFS) with forced growth:
    - Grows `selected` up to target_size (or until no remaining features)
    - Does NOT stop just because the score doesn't improve
    - Tracks the best subset seen at ANY time (global best)
    """
    ALL_FEATURES = load_all_features_from_yaml()
    remaining = ALL_FEATURES.copy()
    selected = []

    # Global best across the whole search (any size)
    best_subset = []
    best_score = -1.0

    # Score of the CURRENT working subset
    current_score = -1.0

    while remaining and len(selected) < target_size:
        # 1) FORWARD STEP: try adding one feature
        best_added = None
        # IMPORTANT: baseline is -inf, thus the argmax candidate is ALWAYS picked
        best_added_score = float("-inf")

        for f in remaining:
            candidate = selected + [f]
            score = run_full_pipeline(combine_mode, candidate)
            if score is None:
                continue

            # Track GLOBAL best (any size)
            if score > best_score:
                best_score = score
                best_subset = candidate.copy()
                print(f"[GLOBAL BEST] {best_subset} (score={best_score:.4f})")

            # Choose best feature to add for THIS step
            if score > best_added_score:
                best_added_score = score
                best_added = f

        # If absolutely nothing worked, then we must stop
        if best_added is None:
            print("\nNo valid candidate to add. Stopping SFFS.")
            break

        # Accept the added feature EVEN IF best_added_score <= current_score
        selected.append(best_added)
        remaining.remove(best_added)
        current_score = best_added_score

        print(f"\n[FORWARD] Selected: {selected} (current_score={current_score:.4f})")

    print("\n" + "#" * 80)
    print(f"FINAL BEST SUBSET for combine='{combine_mode}':")
    print(best_subset)
    print(f"Best score = {best_score:.4f}")
    print("#" * 80)

    return selected, best_score


if __name__ == "__main__":
    sffs_selection(combine_mode=LABEL, target_size=TARGET)
