# Adapted from: TriAD (https://github.com/pseudo-Skye/TriAD).
# Modifications by Thuan Anh Bui, 2025.
# Changes: added multivariate support, updated window slicing,
#          integrated new feature extractor and enabled flexible input configs for testing

import numpy as np
import os, time, datetime, yaml, argparse

import torch
import torch.nn.functional as F

from tqdm import tqdm
from model.tsad import TSAD
from model.losses import ts_loss

from preprocess_data import load_anomaly_smartgrid
from configs.grid_settings import ID, LABEL, MULTIVARIATE

from utils.tsdata import TrainDataset
from utils.utils import find_period, sliding_window, sliding_window_mv, cal_sim, summarize_sim
from utils.utils import save_model, load_model, pkl_save
from utils.transformation import get_cross_domain_features, get_cross_domain_features_mv, get_test_features, get_test_features_mv


def train_one_epoch(net, train_loader, optimizer, alpha, device): 
    n_epoch_iters = 0
    train_loss = 0
    net.train(True)
    for x in train_loader:
        optimizer.zero_grad()
        org_ts, tran_ts, org_fft, tran_fft, org_resid, tran_resid = x[0].to(device), x[1].to(device), x[2].to(device), x[3].to(device), x[4].to(device), x[5].to(device)
        r_org = net(org_ts, org_fft, org_resid) # D * B * T
        r_tran = net(tran_ts, tran_fft, tran_resid)
        loss = ts_loss(r_org, r_tran, alpha=alpha) # D * B * T
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        n_epoch_iters += 1 
    train_loss /= n_epoch_iters
    return train_loss

# The purpose of validation is to maximize the similarity between pos and negatives 
def valid_one_epoch(net, val_features, device): 
    net.train(False)
    batches = val_features[0].shape[0]
    org_repr = []
    tran_repr = []
    for val_i in range(batches):
        org_ts = val_features[0][val_i].unsqueeze(0).to(device)
        tran_ts = val_features[1][val_i].unsqueeze(0).to(device)
        org_fft = val_features[2][val_i].unsqueeze(0).to(device)
        tran_fft = val_features[3][val_i].unsqueeze(0).to(device)
        org_resid = val_features[4][val_i].unsqueeze(0).to(device)
        tran_resid = val_features[5][val_i].unsqueeze(0).to(device)

        org_res = net(org_ts, org_fft, org_resid).detach().cpu() # D x B x T
        tran_res = net(tran_ts, tran_fft, tran_resid).detach().cpu()
        org_repr.append(org_res)
        tran_repr.append(tran_res)
    
    org_repr = torch.cat(org_repr,dim=1).to(torch.float32) # D x all_window x T
    tran_repr = torch.cat(tran_repr,dim=1).to(torch.float32) # D x all_window x T
    sim = cal_sim(org_repr, tran_repr) # D x 2B x 2B
    pos_sim, neg_sim = summarize_sim(sim) # D x B
    dist = pos_sim-neg_sim
    val_dist = dist.mean()
    return val_dist

# Train TriAD on either univariate or multivariate input
def train_dataset(params, train_data, val_data, period_len, device, 
                  model_fn, window_size, stride, multivariate, verbose=False):
    epochs = params['epochs']
    out_dim = params['model_kwargs']['output_dims']
    depth   = params['model_kwargs']['depth']
    alpha   = params['triad_kwargs']['alpha']
    lr      = params['optimizer_kwargs']['lr']
    n_batch = params['batch_size']

    # Build training features
    if multivariate:    # (T, C)
        train_slices = sliding_window_mv(train_data, window_size, stride)  # (B, W, C)
        train_features, _, _ = get_cross_domain_features_mv(
            train_slices, period_len, window_size
        )
    else:   # Only first column (T,)
        train_slices = sliding_window(train_data[:, 0], window_size, stride)  # (B, W)
        train_features, _, _ = get_cross_domain_features(
            train_slices, period_len, window_size
        )

    train_dataset = TrainDataset(train_features)  # org_ts, tran_ts, org_fft, tran_fft, org_resid, tran_resid
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=min(len(train_dataset), n_batch),
        shuffle=False,
        drop_last=True
    )

    # Build validation features
    validation = True
    if len(val_data) < window_size:
        validation = False
    else:
        if multivariate:
            val_slices = sliding_window_mv(val_data, window_size, stride)  # (B, W, C)
            val_features, _, _ = get_cross_domain_features_mv(
                val_slices, period_len, window_size
            )
        else:
            val_slices = sliding_window(val_data[:, 0], window_size, stride)  # (B, W)
            val_features, _, _ = get_cross_domain_features(
                val_slices, period_len, window_size
            )

    # Input dimension depends on uni / multi
    if multivariate:
        C = train_data.shape[1]
        input_dims = C
    else:
        input_dims = 1

    model = TSAD(input_dims=input_dims, output_dims=out_dim, depth=depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)

    # Training loop
    max_val_dist = -1e10
    for epoch in range(0, epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, alpha, device)

        if validation:
            val_dist = valid_one_epoch(model, val_features, device)
            if verbose:
                print(
                    f'Epoch #{epoch}: Training loss: {train_loss} '
                    f'\t\t Validation distance (distance between pos and neg): {val_dist}'
                )

            if max_val_dist < val_dist:
                if verbose:
                    print(
                        f'Validation Distance Increased({max_val_dist:.6f}'
                        f'--->{val_dist:.6f}) \t Saving The Model'
                    )
                max_val_dist = val_dist
                save_model(model, model_fn)

    if not validation or max_val_dist == -1e10:
        save_model(model, model_fn)

# Run TriAD on either univariate or multivariate test data
def test_dataset(params, test_data, period_len, device, model_fn, 
                 window_size, stride, multivariate, verbose=False):
    out_dim = params['model_kwargs']['output_dims']
    depth   = params['model_kwargs']['depth']

    # 1. Build test features
    if multivariate:
        test_slices = sliding_window_mv(test_data, window_size, stride)  # (B, W, C)
        test_ft = get_test_features_mv(test_slices, period_len)
        input_dims = test_data.shape[1]
    else:
        test_slices = sliding_window(test_data[:, 0], window_size, stride)  # (B, W)
        test_ft     = get_test_features(test_slices, period_len)
        input_dims  = 1

    # Load model
    model = TSAD(input_dims=input_dims, output_dims=out_dim, depth=depth).to(device)
    if os.path.exists(model_fn):
        if verbose:
            print(f"[INFO] Resuming from {model_fn}")
        load_model(model, model_fn, device)
    else:
        if verbose:
            print("[INFO] Training from scratch (no resume checkpoint found)")

    # Scores
    scores = test_eval(model, test_ft, device)
    return scores

# The test evaluation returns the scores of similarity of each window to the others 
def test_eval(model, test_ft, device):
    model.eval()
    batches = test_ft[0].shape[0]
    repr = []
    for test_i in range(batches):
        org_ts = test_ft[0][test_i].unsqueeze(0).to(device)
        org_fft = test_ft[1][test_i].unsqueeze(0).to(device)
        org_res = test_ft[2][test_i].unsqueeze(0).to(device)
        res = model(org_ts, org_fft, org_res).detach().cpu() # D x B x T
        repr.append(res)
    
    repr = torch.cat(repr,dim=1).to(torch.float32)
    z = F.normalize(repr, p=2, dim=2)
    sim = torch.abs(torch.matmul(z, z.transpose(1, 2))) # D x B x B
    # Remove the similarity between instance itself
    sim_updated = torch.tril(sim, diagonal=-1)[:, :, :-1]    # D x B x (B-1)
    sim_updated += torch.triu(sim, diagonal=1)[:, :, 1:] 
    scores = sim_updated.mean(dim=-1).numpy()
    return scores

def compute_window_and_stride(period_len, cycles, stride_ratio):
    window_size = max(1, round(cycles * period_len))
    stride = max(1, window_size // stride_ratio)
    return window_size, stride


def main(cfg_path: str):
    torch.cuda.empty_cache()
    torch.cuda.manual_seed_all(0)
    device = torch.device('cuda')

    with open(cfg_path, "r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f)

    id = ID
    drop_10 = True
    dataset_kwargs = CFG["dataset_kwargs"]
    label_cfg = dataset_kwargs["labels"]

    print("\n" + "=" * 80)
    print(f"TRAIN RUN: combine='{LABEL}' | features={dataset_kwargs['feature_columns']}")
    print("=" * 80)

    # Load data with this specific feature subset
    train_x, valid_x, test_x, test_y = load_anomaly_smartgrid(
        dataset_kwargs["csv_path"],
        key=dataset_kwargs["key"],
        label=label_cfg["columns"],
        combine=label_cfg["combine"],
        feature_cols=dataset_kwargs["feature_columns"],
    )

    train_data  = train_x[id]
    val_data    = valid_x[id]
    test_data   = test_x[id]
    test_labels = test_y[id]

    # Folders (optional: per combination/feature, or all in one)
    run_dir     = "trained"
    results_dir = "./results"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Unique filenames per setting
    model_fn     = f"{run_dir}/model.pkl"
    tri_res_path = f"{results_dir}/tri_res.pt"

    period_len = find_period(train_data[:, 0], id)

    if drop_10:
        train_data = train_data[len(train_data) // 10:]

    cycles  = CFG["triad_kwargs"]["cycles"]
    ratio   = CFG["triad_kwargs"]["stride_ratio"]

    # Windowing and stride
    window_size, stride = compute_window_and_stride(period_len, cycles, ratio)

    # Training and Validating
    train_dataset(CFG, train_data, val_data, period_len, device, model_fn=model_fn, 
                window_size=window_size, stride=stride, multivariate=MULTIVARIATE)
    
    # Testing
    scores = test_dataset(CFG, test_data, period_len, device, model_fn=model_fn, 
                        window_size=window_size, stride=stride, multivariate=MULTIVARIATE)
    
    # Typically 3 score sequences: obs, freq, resid
    obs_anom  = int(np.argmin(scores[0]))
    freq_anom = int(np.argmin(scores[1]))
    res_anom  = int(np.argmin(scores[2]))
    suspects  = np.unique(np.array([obs_anom, freq_anom, res_anom], dtype=int))

    # Map windows back to time indices & labels (still univariate)
    label_slices = sliding_window(test_labels.astype(int), window_size, stride)
    index_slices = sliding_window(np.arange(len(test_data)), window_size, stride)

    if label_slices.size == 0 or index_slices.size == 0:
        raise ValueError(
            f"No windows produced. window_size={window_size}, "
            f"stride={stride}, len(test_data)={len(test_data)}."
        )

    anom_win = np.where(np.any(label_slices == 1, axis=1))[0]
    is_within_anom = np.any(np.isin(suspects, anom_win)) if anom_win.size > 0 else False

    res_notebook = {
        "id": id,
        "combine_mode": LABEL,
        "tri_detected": bool(is_within_anom),
        "num_suspects": int(len(suspects)),
        "suspects": [
            (int(index_slices[i, 0]), int(index_slices[i, -1])) for i in suspects
        ]
    }
    tqdm.write(
        f"{LABEL} | {id}: anomaly DETECTED"
        if is_within_anom else
        f"{LABEL} | {id}: anomaly MISS"
    )
    pkl_save(tri_res_path, res_notebook)


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
