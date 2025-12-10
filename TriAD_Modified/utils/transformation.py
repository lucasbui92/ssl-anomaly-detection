import numpy as np
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import torch, random
from utils.utils import normalize_arr_2d, robust_scaling_2d


def robust_scaling_3d(arr_3d):
    """
    Apply robust_scaling_2d channel-wise to a (B, T, C) array.
    Assumes robust_scaling_2d(X) expects shape (n_features, n_timesteps).
    """
    B, T, C = arr_3d.shape
    # reshape to (C, B*T) so each channel is a "feature row"
    reshaped = arr_3d.reshape(-1, C).T              # (C, B*T)
    scaled = robust_scaling_2d(reshaped)            # (C, B*T)
    return scaled.T.reshape(B, T, C)                # (B, T, C)

def gen_jitering(ts, window_size):
    num_samples = len(ts)
    noise_amplitude = (ts.max()-ts.min())*1/5  # Amplitude of the noise signal
    # Combine the original time series with the high frequency noise
    combined_signal = ts + noise_amplitude * np.random.randn(num_samples)
    # Random length of the frequency-based anomalies
    anom_len = np.random.randint(window_size//20, window_size//3)
    # Randomly select the range of indices for the part to modify
    start_index = np.random.randint(0, len(ts)-anom_len)
    end_index = start_index+anom_len
    # Revise portion of the original time series
    modified_ts = ts.copy()
    modified_ts[start_index:end_index] = combined_signal[start_index:end_index]

    return modified_ts, (start_index,end_index)

def gen_warping(ts, fft_values, window_size, verbose = False):
    # Compute the power spectral density
    psd_values = np.abs(fft_values) ** 2
    # Find the peak 30 frequencies
    peak_indices = np.argsort(psd_values)[-30:]

    frequencies = np.fft.fftfreq(len(ts), d=1)
    frequencies = frequencies[peak_indices]
    # Get the positive frequencies between (0,1)
    frequencies = np.unique(frequencies[frequencies>0])
    frequencies = np.sort(frequencies) # frequency sorted from lowest to highest

    # Randomly pick frequencies from the lower frequency range  
    pick_idx = np.arange(0, len(frequencies), 3)
    cutoff = np.random.choice(frequencies[pick_idx][0:4], size=2, replace=False)
    # Randomly select the frequency range you want to enhance
    low_freq = min(cutoff)  
    high_freq = max(cutoff)  
    b, a = signal.butter(4, [low_freq, high_freq], btype='band')
    # Apply the filter to the time series
    filtered_signal_lower = signal.lfilter(b, a, ts)

    # Scale the filtered signal to the orignal time series
    original = ts.reshape(-1, 1)
    filtered_signal_lower = filtered_signal_lower.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(original.min(), original.max()))
    scaler.fit(original)
    filtered_signal_lower = scaler.transform(filtered_signal_lower).flatten()

    # Random length of the frequency-based anomalies
    anom_len = np.random.randint(window_size//20, window_size//3)
    # Randomly select the range of indices for the part to modify
    start_index = np.random.randint(0, len(ts)-anom_len)
    end_index = start_index+anom_len

    # Copy the original array
    modified_ts = ts.copy()
    modified_ts[start_index:end_index] = filtered_signal_lower[start_index:end_index]
    if verbose:
        print(f'Lower frequency transformation: filter between {low_freq} and {high_freq}, anomaly length: {anom_len}')
    
    return modified_ts, (start_index,end_index)

def get_cross_domain_features(ts_slices, period_len, window_size):
    random.seed()
    tran_ts = []
    org_fft = []
    tran_fft = []
    org_resid = []
    tran_resid = []
    anom_indx = []
    h_freq_num = 0
    jittering = True
    for slice in ts_slices:
        # 1. Get the residuals - original
        org_result = seasonal_decompose(slice, model='additive', period=period_len, extrapolate_trend='freq')
        org_resid.append(org_result.resid)
        # 2. Get the frequency domain features - original
        fft_values = np.fft.fft(slice)
        amplitudes = np.abs(fft_values)  # Magnitude spectrum
        phases = np.angle(fft_values)  # Phase spectrum
        power = np.abs(fft_values) ** 2
        cat_fft = np.vstack((amplitudes, phases, power))
        cat_fft = robust_scaling_2d(cat_fft)  # 3*each row
        org_fft.append(cat_fft)
        
        #--------------------------------------------------
        # Transform the input time series (jittering and smoothing take turns)
        if jittering:
            modified, anom = gen_jitering(slice, window_size)
            h_freq_num += 1
            jittering = False
        else:
            modified, anom = gen_warping(slice, fft_values, window_size, verbose=False)
            jittering = True

        tran_ts.append(modified)
        anom_indx.append(anom)
        # 1. Get the residuals - transformed
        tran_result = seasonal_decompose(modified, model='additive', period=period_len,  extrapolate_trend='freq')
        tran_resid.append(tran_result.resid)

        # 2. Get the fft features - transformed
        fft_values = np.fft.fft(modified)
        amplitudes = np.abs(fft_values)  # Magnitude spectrum
        phases = np.angle(fft_values)  # Phase spectrum
        power = np.abs(fft_values) ** 2
        cat_fft = np.vstack((amplitudes, phases, power))
        cat_fft = robust_scaling_2d(cat_fft)  # 3*each row
        tran_fft.append(cat_fft)        

    slices = robust_scaling_2d(ts_slices)
    org_resid = robust_scaling_2d(org_resid)
    tran_ts = robust_scaling_2d(tran_ts)
    tran_resid = robust_scaling_2d(tran_resid)

    org_ts = torch.Tensor(slices).unsqueeze(dim=-1) # B*T*1
    org_fft = torch.Tensor(np.array(org_fft)).permute(0,2,1) # B*T*3
    org_resid = torch.Tensor(np.array(org_resid)).unsqueeze(dim=-1) # B*T*1

    tran_ts = torch.Tensor(np.array(tran_ts)).unsqueeze(dim=-1)
    tran_fft = torch.Tensor(np.array(tran_fft)).permute(0,2,1) 
    tran_resid = torch.Tensor(np.array(tran_resid)).unsqueeze(dim=-1)
    features = [org_ts, tran_ts, org_fft, tran_fft, org_resid, tran_resid]
    return features, h_freq_num, anom_indx

def get_cross_domain_features_mv(ts_slices, period_len, window_size):
    """
    Multivariate version of get_cross_domain_features.

    ts_slices: (B, W, C)
    """
    random.seed()
    ts_slices = np.asarray(ts_slices)
    B, W, C = ts_slices.shape

    tran_ts_list   = []
    org_fft_list   = []
    tran_fft_list  = []
    org_resid_list = []
    tran_resid_list= []
    anom_indx      = []

    h_freq_num = 0
    jittering = True

    for b in range(B):
        slice_mv = ts_slices[b]          # (W, C)

        # Residuals (original) per channel
        resid_channels = []
        for c in range(C):
            series = slice_mv[:, c].astype(float)  # ensure 1D float
            if W < 2 * period_len:
                # Fallback: simple de-mean if window too short for seasonal_decompose
                resid = series - np.nanmean(series)
            else:
                org_result = seasonal_decompose(
                    series,
                    model="additive",
                    period=period_len,
                    extrapolate_trend="freq",
                )
                resid = org_result.resid
            resid_channels.append(resid)
        resid_channels = np.stack(resid_channels, axis=-1)  # (W, C)
        org_resid_list.append(resid_channels)

        # FFT features (original) per channel
        fft_feats_channels = []
        for c in range(C):
            series = slice_mv[:, c].astype(float)
            fft_values = np.fft.fft(series)
            amplitudes = np.abs(fft_values)
            phases     = np.angle(fft_values)
            power      = np.abs(fft_values) ** 2
            cat_fft_c  = np.stack([amplitudes, phases, power], axis=-1)  # (W, 3)
            fft_feats_channels.append(cat_fft_c)

        fft_feats_channels = np.stack(fft_feats_channels, axis=1)  # (W, C, 3)
        cat_fft = fft_feats_channels.reshape(W, 3 * C)             # (W, 3C)
        cat_fft = cat_fft.T                                       # (3C, W)
        cat_fft = robust_scaling_2d(cat_fft)                      # (3C, W)
        cat_fft = cat_fft.T                                       # (W, 3C)
        org_fft_list.append(cat_fft)

        # Jittering / warping per channel
        modified_channels = []
        anom_channels     = []
        if jittering:
            for c in range(C):
                series = slice_mv[:, c].astype(float)
                modified_1d, anom_c = gen_jitering(series, window_size)
                modified_channels.append(modified_1d)
                anom_channels.append(anom_c)
            h_freq_num += 1
            jittering = False
        else:
            for c in range(C):
                series = slice_mv[:, c].astype(float)
                fft_values = np.fft.fft(series)
                modified_1d, anom_c = gen_warping(
                    series,
                    fft_values,
                    window_size,
                    verbose=False,
                )
                modified_channels.append(modified_1d)
                anom_channels.append(anom_c)
            jittering = True

        modified = np.stack(modified_channels, axis=-1)  # (W, C)
        tran_ts_list.append(modified)
        anom_indx.append(anom_channels)

        # Residuals (transformed)
        tran_resid_channels = []
        for c in range(C):
            series = modified[:, c].astype(float)
            if W < 2 * period_len:
                resid = series - np.nanmean(series)
            else:
                tran_result = seasonal_decompose(
                    series,
                    model="additive",
                    period=period_len,
                    extrapolate_trend="freq",
                )
                resid = tran_result.resid
            tran_resid_channels.append(resid)
        tran_resid_channels = np.stack(tran_resid_channels, axis=-1)  # (W, C)
        tran_resid_list.append(tran_resid_channels)

        # FFT (transformed)
        fft_feats_channels_tr = []
        for c in range(C):
            series = modified[:, c].astype(float)
            fft_values = np.fft.fft(series)
            amplitudes = np.abs(fft_values)
            phases     = np.angle(fft_values)
            power      = np.abs(fft_values) ** 2
            cat_fft_c  = np.stack([amplitudes, phases, power], axis=-1)
            fft_feats_channels_tr.append(cat_fft_c)

        fft_feats_channels_tr = np.stack(fft_feats_channels_tr, axis=1)  # (W, C, 3)
        cat_fft_tr = fft_feats_channels_tr.reshape(W, 3 * C)             # (W, 3C)
        cat_fft_tr = cat_fft_tr.T                                        # (3C, W)
        cat_fft_tr = robust_scaling_2d(cat_fft_tr)                       # (3C, W)
        cat_fft_tr = cat_fft_tr.T                                        # (W, 3C)
        tran_fft_list.append(cat_fft_tr)

    # Stack & scale
    org_ts    = ts_slices.copy()                    # (B, W, C)
    org_resid = np.array(org_resid_list)           # (B, W, C)
    tran_ts   = np.array(tran_ts_list)             # (B, W, C)
    tran_resid= np.array(tran_resid_list)          # (B, W, C)
    org_fft   = np.array(org_fft_list)             # (B, W, 3C)
    tran_fft  = np.array(tran_fft_list)            # (B, W, 3C)

    org_ts    = robust_scaling_3d(org_ts)
    org_resid = robust_scaling_3d(org_resid)
    tran_ts   = robust_scaling_3d(tran_ts)
    tran_resid= robust_scaling_3d(tran_resid)

    org_ts    = torch.Tensor(org_ts)               # B*T*C
    tran_ts   = torch.Tensor(tran_ts)
    org_fft   = torch.Tensor(org_fft)              # B*T*(3C)
    tran_fft  = torch.Tensor(tran_fft)
    org_resid = torch.Tensor(org_resid)            # B*T*C
    tran_resid= torch.Tensor(tran_resid)

    features = [org_ts, tran_ts, org_fft, tran_fft, org_resid, tran_resid]
    return features, h_freq_num, anom_indx

def get_test_features(ts_slices, period_len):
    org_fft = []
    org_resid = []
    
    for slice in ts_slices:
        # 1. Get the residuals - original
        org_result = seasonal_decompose(slice, model='additive', period=period_len, extrapolate_trend='freq')
        org_resid.append(org_result.resid)
        # 2. Get the fft features - original
        fft_values = np.fft.fft(slice)
        amplitudes = np.abs(fft_values)  # Magnitude spectrum
        phases = np.angle(fft_values)  # Phase spectrum
        power = np.abs(fft_values) ** 2
        cat_fft = np.vstack((amplitudes, phases, power))
        cat_fft = robust_scaling_2d(cat_fft)  # 3*each row
        org_fft.append(cat_fft)

    slices = robust_scaling_2d(ts_slices)
    org_resid = robust_scaling_2d(org_resid)

    org_ts = torch.Tensor(slices).unsqueeze(dim=-1)
    org_fft = torch.Tensor(np.array(org_fft)).permute(0,2,1) 
    org_resid = torch.Tensor(np.array(org_resid)).unsqueeze(dim=-1)

    features = [org_ts, org_fft, org_resid]
    return features 

def get_test_features_mv(ts_slices, period_len):
    ts_slices = np.asarray(ts_slices)
    B, W, C = ts_slices.shape

    org_resid_list = []
    org_fft_list   = []

    for b in range(B):
        slice_mv = ts_slices[b]  # (W, C)

        # 1. Residuals per channel
        resid_channels = []
        for c in range(C):
            org_result = seasonal_decompose(
                slice_mv[:, c],
                model='additive',
                period=period_len,
                extrapolate_trend='freq'
            )
            resid_channels.append(org_result.resid)
        resid_channels = np.stack(resid_channels, axis=-1)  # (W, C)
        org_resid_list.append(resid_channels)

        # 2. FFT features per channel
        fft_feats_channels = []
        for c in range(C):
            fft_values = np.fft.fft(slice_mv[:, c])
            amplitudes = np.abs(fft_values)
            phases     = np.angle(fft_values)
            power      = np.abs(fft_values) ** 2
            # (W, 3) for this channel
            cat_fft_c = np.stack([amplitudes, phases, power], axis=-1)
            fft_feats_channels.append(cat_fft_c)

        # (W, C, 3) -> (W, 3C)
        fft_feats_channels = np.stack(fft_feats_channels, axis=1)  # (W, C, 3)
        cat_fft = fft_feats_channels.reshape(W, 3 * C)             # (W, 3C)
        cat_fft = cat_fft.T                                        # (3C, W)
        cat_fft = robust_scaling_2d(cat_fft)                       # (3C, W)
        cat_fft = cat_fft.T                                        # (W, 3C)
        org_fft_list.append(cat_fft)

    # Stack arrays
    org_ts    = ts_slices.copy()                 # (B, W, C)
    org_resid = np.array(org_resid_list)         # (B, W, C)
    org_fft   = np.array(org_fft_list)           # (B, W, 3*C)

    # Robust scaling on time-domain and residuals
    org_ts    = robust_scaling_3d(org_ts)
    org_resid = robust_scaling_3d(org_resid)
    # FFT already scaled per-window above

    # Convert to tensors
    org_ts    = torch.Tensor(org_ts)             # B * T * C
    org_fft   = torch.Tensor(org_fft)            # B * T * (3C)
    org_resid = torch.Tensor(org_resid)          # B * T * C

    features = [org_ts, org_fft, org_resid]
    return features
