import argparse
import os
import numpy as np
import pandas as pd
import joblib
from scipy.io import wavfile

CLASS_NAMES = ['Sine Wave', 'FM Signal', 'AM Signal', 'Pulse Signal', 'Noise']


def load_time_signal(path: str, channel: int = 0) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.wav':
        sr, data = wavfile.read(path)
        if data.ndim > 1:
            if channel < 0 or channel >= data.shape[1]:
                raise ValueError(f'Invalid channel {channel} for WAV with {data.shape[1]} channels')
            data = data[:, channel]
        if np.issubdtype(data.dtype, np.integer):
            maxv = np.iinfo(data.dtype).max
            data = data.astype(np.float32) / float(maxv)
        else:
            data = data.astype(np.float32)
        return data
    elif ext == '.npy':
        arr = np.load(path)
        return np.asarray(arr).astype(np.float32).ravel()
    elif ext == '.csv':
        df = pd.read_csv(path, header=None)
        if df.shape[0] == 1:
            sig = df.iloc[0].values
        elif df.shape[1] == 1:
            sig = df.iloc[:, 0].values
        else:
            sig = df.values.ravel()
        return np.asarray(sig, dtype=np.float32)
    else:
        raise ValueError(f'Unsupported input type: {ext}. Use .wav, .npy, or .csv')


def compute_fft_features(signal: np.ndarray, n_bins: int = 1024, window: str = 'hann') -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    x = x - np.mean(x)
    if window == 'hann':
        w = np.hanning(len(x))
        x = x * w
    if len(x) < n_bins:
        pad = np.zeros(n_bins - len(x), dtype=x.dtype)
        x = np.concatenate([x, pad], axis=0)
    # No truncation before FFT to preserve low-frequency resolution when input is long
    X = np.fft.fft(x)
    mag = np.abs(X)[:n_bins]
    return mag.astype(np.float32)


def predict(features: np.ndarray, model_path: str):
    model = joblib.load(model_path)
    probs = model.predict_proba(features.reshape(1, -1))[0]
    idx = int(np.argmax(probs))
    name = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else str(idx)
    conf = float(probs[idx])
    return idx, name, conf, probs


def main():
    parser = argparse.ArgumentParser(description='Infer class from a time-domain signal by FFT -> 1024-bin magnitude -> RF model')
    parser.add_argument('--input', required=True, help='Path to time-domain signal (.wav, .npy, or .csv)')
    parser.add_argument('--model', default='models/rf_model.pkl', help='Path to saved RandomForest model (joblib)')
    parser.add_argument('--window', choices=['hann', 'none'], default='hann', help='Window to apply before FFT')
    parser.add_argument('--channel', type=int, default=0, help='Channel index for multi-channel WAV')
    parser.add_argument('--save', type=str, default='', help='Optional path to save features CSV (1024 bins)')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f'Error: model not found at {args.model}. Train with: python train.py')
        return
    if not os.path.exists(args.input):
        print(f'Error: input not found at {args.input}')
        return

    sig = load_time_signal(args.input, channel=args.channel)
    win = None if args.window == 'none' else 'hann'
    feats = compute_fft_features(sig, n_bins=1024, window=win)

    idx, name, conf, probs = predict(feats, args.model)

    print('\nPrediction:')
    print('-----------')
    print(f'Class: {idx} ({name})')
    print(f'Confidence: {conf:.4f}')

    if args.save:
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        pd.DataFrame(feats.reshape(1, -1)).to_csv(args.save, index=False)
        print(f'Saved 1024-bin FFT magnitude features to {os.path.abspath(args.save)}')

if __name__ == '__main__':
    main()
