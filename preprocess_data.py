import numpy as np
import pandas as pd


# Labels: default to 0 if missing; cast to {0,1}
def to01(arrlike):
    return pd.Series(arrlike).fillna(0).astype(int).clip(0, 1).to_numpy()

def load_anomaly_smartgrid(csv_path, key, label, combine, feature_cols):
    """
    Returns:
        train_x, valid_x, test_x, test_y
        where each is a dict { key : array }

    X shape: (T, C)
    y shape: (T,)
    """
    df = pd.read_csv(csv_path).sort_values("Timestamp")

    # Label handling
    if isinstance(label, str):
        label_cols = [label]
    elif isinstance(label, (list, tuple)) and len(label) == 2:
        label_cols = list(label)
    else:
        raise ValueError("label must be a string or a 2-list.")

    # Build X
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = df[feature_cols].to_numpy(np.float32)

    # Compute a single combined label y
    if len(label_cols) == 1:
        y = to01(df[label_cols[0]])
    else:
        y_over = to01(df[label_cols[0]])
        y_tran = to01(df[label_cols[1]])

        if combine == "any":
            y = ((y_over == 1) | (y_tran == 1)).astype(np.int64)
        elif combine == "both":
            y = ((y_over == 1) & (y_tran == 1)).astype(np.int64)
        elif combine == "overload":
            y = y_over.astype(np.int64)
        elif combine == "transformer":
            y = y_tran.astype(np.int64)
        else:
            raise ValueError(f"Unknown combine='{combine}'.")

    # Temporal split (70/10/20)
    n_tr = int(0.7 * len(X))
    n_va = int(0.1 * len(X))
    X_train, X_valid, X_test = X[:n_tr], X[n_tr:n_tr+n_va], X[n_tr+n_va:]
    y_train, y_valid, y_test = y[:n_tr], y[n_tr:n_tr+n_va], y[n_tr+n_va:]

    train_x = {key: X_train.astype(np.float32)}
    valid_x = {key: X_valid.astype(np.float32)}
    test_x  = {key: X_test.astype(np.float32)}

    # Only test labels needed downstream
    test_y  = {key: y[n_tr+n_va:].astype(np.int64)}

    return train_x, valid_x, test_x, test_y
