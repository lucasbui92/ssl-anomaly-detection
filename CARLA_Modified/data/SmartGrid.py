# Created by: Thuan Anh Bui (2025)
# Description: Implemented Smart Grid dataset preprocessing, windowing technqiues
#               and enabled decision boundary for anomaly.

import os, yaml
import numpy as np
import pandas as pd
import torch
from utils.mypath import MyPath
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SmartGrid(Dataset):
    """Smart Grid Monitoring Dataset

    Args:
        fname (string): Base filename without .csv extension.
        root (string): Dataset root folder (default via MyPath.db_root_dir('smart_grid')).
        train (bool): True for training set, False for test set.
        transform (callable, optional): Optional transform for each timeseries.
        sanomaly: (unused placeholder for compatibility)
        mean_data (np.ndarray, optional): Mean of training data for normalization.
        std_data (np.ndarray, optional): Std of training data for normalization.
        dual_binary (bool): False -> Dataset using 'Fault Indicator';
                           True  -> Dataset using 'Overload Condition' and 'Transformer Fault'
    """
    base_folder = ''

    def __init__(self, fname, root=MyPath.db_root_dir('smart_grid'), train=True, transform=None, sanomaly=None,
                 mean_data=None, std_data=None, dual_binary=False):

        super(SmartGrid, self).__init__()
        self.root = root
        self.transform = transform
        self.sanomaly = sanomaly
        self.train = train
        self.dual_binary = dual_binary
        self.classes = ['Normal', 'Anomaly']

        self.data = []
        self.targets = []

        # Load the single file 
        file_path = os.path.join(self.root, fname)
        if not file_path.endswith('.csv'):
            file_path += '.csv'
        df = pd.read_csv(file_path)

        # Load the config file
        config_loc = yaml.safe_load(open("configs/smartgrid.yml"))
        
        # --- Dataset (Major) ---
        cfg_major  = config_loc["smart_grid_major"]
        ACTIVE_FEATURES = cfg_major["feature_columns"]
        LABEL_MODE      = cfg_major["label_mode"]

        # Get values for window size and threshold
        window_cfg = cfg_major["window"]
        self.w_size          = window_cfg["size"]
        self.stride          = window_cfg["stride"]
        threshold = cfg_major["threshold"]
        self.threshold_mode  = threshold["mode"]
        self.threshold_value = threshold["value"]

        if df.isna().to_numpy().sum() > 0:
            print('Data contains NaN which replaced with zero')
            df = df.fillna(0)

        # Build labels depending on dataset
        if not dual_binary:     # Single column 'Fault Indicator' in {0,1,2}
            if 'Fault Indicator' not in df.columns:
                raise ValueError(f"'Fault Indicator' column not found in {file_path} (dual_binary=False).")
            fault = df['Fault Indicator'].astype(int).to_numpy()
            labels = (fault > 0).astype(int)  # (1 or 2) -> 1
            label_cols_to_drop = ['Fault Indicator']
        else:
            label_cols = ['Overload Condition', 'Transformer Fault']    # Two binary columns
            missing = [c for c in label_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns {missing} in {file_path} (dual_binary=True).")
            overload = df['Overload Condition'].astype(int).to_numpy()
            tfault   = df['Transformer Fault'].astype(int).to_numpy()

            if LABEL_MODE == "any":
                labels = ((overload == 1) | (tfault == 1)).astype(int)
            elif LABEL_MODE == "both":
                labels = ((overload == 1) & (tfault == 1)).astype(int)
            elif LABEL_MODE == "overload_only":
                labels = (overload == 1).astype(int)
            elif LABEL_MODE == "tfault_only":
                labels = (tfault == 1).astype(int)
            else:
                raise ValueError(f"Unknown label_mode '{LABEL_MODE}'")
            label_cols_to_drop = label_cols

        # Feature engineering
        drop_cols = label_cols_to_drop + (['Timestamp'] if 'Timestamp' in df.columns else [])
        feature_df = df.drop(columns=drop_cols, errors='ignore')

        missing = [c for c in ACTIVE_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Select features
        feature_df = df[ACTIVE_FEATURES].copy()
        features = feature_df.to_numpy().astype(np.float32) 

        # Compute + apply normalization *before* splitting
        if (mean_data is not None) and (std_data is not None):
            self.mean = mean_data
            self.std  = std_data
        else:
            normal_mask = (labels == 0)
            pool = features[normal_mask] if np.any(normal_mask) else features
            self.mean = np.mean(pool, axis=0)
            self.std  = np.std(pool, axis=0)
            self.std[self.std == 0.0] = 1.0

        features = (features - self.mean) / self.std

        # Time-based split on normalized features
        n_samples = len(features)
        split_idx = int(0.7 * n_samples)

        if self.train:
            split_features = features[:split_idx]
            split_labels   = labels[:split_idx]

            # Apply augmentations on *normalized* features
            split_features = self.apply_augmentations(split_features, config_loc["augment"])
        else:
            split_features = features[split_idx:]
            split_labels   = labels[split_idx:]

        self.data    = np.asarray(split_features, dtype=np.float32)
        self.targets = np.asarray(split_labels,   dtype=np.int64)
        self.data, self.targets = self.convert_to_windows()  

    def convert_to_windows(self):
        N = self.data.shape[0]
        if N < self.w_size:
            raise ValueError(f"Series length {N} is smaller than window size {self.w_size}.")

        windows = []
        wlabels = []
        # fractions = []
        sz = (N - self.w_size) // self.stride + 1

        for i in range(sz):
            st = i * self.stride
            ed = st + self.w_size
            window = self.data[st:ed]

            anomalous_points = (self.targets[st:ed] > 0).sum()
            if self.threshold_mode == "fraction":
                frac = anomalous_points / float(self.w_size)
                lbl = 1 if frac >= self.threshold_value else 0
                # fractions.append(frac)

            elif self.threshold_mode == "count":
                lbl = 1 if anomalous_points >= self.threshold_value else 0

            else:
                raise ValueError(f"Unknown threshold_mode '{self.threshold_mode}'")

            windows.append(window)
            wlabels.append(lbl)

        # TODO: Use this output to confirm that the threshold is high enough for evaluation of the model
        print("Window label distribution:", np.unique(wlabels, return_counts=True))
        # print("Fraction stats: min=", min(fractions), "max=", max(fractions))

        return np.stack(windows, axis=0), np.array(wlabels, dtype=np.int64)
    
    def apply_augmentations(self, X, cfg_aug):
        """
        Apply time-series augmentations defined in the YAML config.
        X: numpy array of shape (T, C)
        cfg_aug: dictionary loaded from YAML under "augment"
        """
        X_aug = X.astype(np.float32).copy()
        T, C = X_aug.shape

        #  Gaussian Noise
        if "gaussian" in cfg_aug:
            g = cfg_aug["gaussian"]
            if np.random.rand() < float(g["p"]):
                std = float(g["std"])
                X_aug += np.random.normal(0, std, X_aug.shape).astype(np.float32)

        #  Jitter
        if "jitter" in cfg_aug:
            j = cfg_aug["jitter"]
            if np.random.rand() < float(j["p"]):
                std = float(j["std"])
                X_aug += np.random.normal(0, std, X_aug.shape).astype(np.float32)

        #  Time Masking
        if "time_mask" in cfg_aug:
            tmask = cfg_aug["time_mask"]
            if np.random.rand() < float(tmask["p"]):
                max_len = int(float(tmask["max_mask_frac"]) * T)
                mask_len = np.random.randint(1, max_len + 1)
                start = np.random.randint(0, T - mask_len + 1)
                X_aug[start:start + mask_len, :] = 0.0

        #  Cropping
        if "cropping" in cfg_aug:
            cr = cfg_aug["cropping"]
            if np.random.rand() < float(cr["p"]):
                min_frac = float(cr["min_frac"])
                crop_len = max(2, int(min_frac * T))
                start = np.random.randint(0, T - crop_len + 1)
                cropped = X_aug[start:start+crop_len, :]
                X_aug = self._resample(cropped, T)

        #  Time Scaling
        if "time_scaling" in cfg_aug:
            sc = cfg_aug["time_scaling"]
            if np.random.rand() < float(sc["p"]):
                scale = np.random.uniform(float(sc["min"]), float(sc["max"]))
                new_len = max(2, int(T * scale))
                scaled = self._resample(X_aug, new_len)
                X_aug = self._resample(scaled, T)
        return X_aug
    
    # Helper for scaling + cropping
    def _resample(self, x, new_len):
        old_T, C_ = x.shape
        old_idx = np.arange(old_T)
        new_idx = np.linspace(0, old_T - 1, new_len)
        out = np.zeros((new_len, C_), dtype=np.float32)
        for c in range(C_):
            out[:, c] = np.interp(new_idx, old_idx, x[:, c])
        return out

    def __getitem__(self, index):
        ts_org = torch.from_numpy(self.data[index]).float().to(device)
        target_val = int(self.targets[index])  # scalar python int
        target = torch.tensor(target_val, dtype=torch.long).to(device)
        ts_size = (ts_org.shape[0], ts_org.shape[1])
        class_name = self.classes[target_val]  # use int, not tensor, as index
        return {
            'ts_org': ts_org,
            'target': target,
            'meta': {'ts_size': ts_size, 'index': index, 'class_name': class_name}
        }

    def __len__(self):
        return len(self.data)

    def get_ts(self, index):
        return self.data[index]

    def get_info(self):
        return self.mean, self.std

    def concat_ds(self, new_ds):
        self.data = np.concatenate((self.data, new_ds.data), axis=0)
        self.targets = np.concatenate((self.targets, new_ds.targets), axis=0)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train else "Test")
