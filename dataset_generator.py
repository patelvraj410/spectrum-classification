import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


# ------------------------------------------------------------
# SIGNAL GENERATORS
# ------------------------------------------------------------

def sine_wave(freq, sr, dur, noise=0.02):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return np.sin(2*np.pi*freq*t) + np.random.normal(0, noise, len(t))


def fm_signal(freq, sr, dur, mod_freq=300, mod_index=5, noise=0.02):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return np.sin(2*np.pi*freq*t + mod_index*np.sin(2*np.pi*mod_freq*t)) + np.random.normal(0, noise, len(t))


def am_signal(freq, sr, dur, mod_freq=200, mod_index=0.7, noise=0.02):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    carrier = np.sin(2*np.pi*freq*t)
    mod = 1 + mod_index*np.sin(2*np.pi*mod_freq*t)
    return carrier * mod + np.random.normal(0, noise, len(t))


def pulse_signal(freq, sr, dur, duty=0.2, noise=0.02):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sq = signal.square(2*np.pi*freq*t, duty=duty)
    return sq + np.random.normal(0, noise, len(t))


def white_noise(sr, dur, noise=1.0):
    return np.random.normal(0, noise, int(sr*dur))


# ------------------------------------------------------------
# MAIN DATASET GENERATOR
# ------------------------------------------------------------

def create_dataset(
    num_samples=2000,
    sr=20000,             # PERFECT sample rate (20 kHz)
    dur=0.1,              # 0.1s → 2000 samples
    num_features=1024,
    out_csv="data/rf_spectrum_dataset.csv",
):
    """
    Generates PERFECT separable RF spectra:
      - Sine
      - FM
      - AM
      - Pulse
      - Noise
    """
    os.makedirs("data", exist_ok=True)

    N = int(sr * dur)
    if num_features > N:
        raise ValueError("num_features > sample_length, reduce num_features")

    # CLEAR, SEPARATED FREQUENCIES
    class_freqs = {
        0: 1000,   # 1 kHz
        1: 3000,   # 3 kHz
        2: 5000,   # 5 kHz
        3: 7000,   # 7 kHz
        4: None
    }

    class_names = [
        "Sine Wave",
        "FM Signal",
        "AM Signal",
        "Pulse Signal",
        "Noise"
    ]

    X = []
    y = []

    print("\nGenerating dataset:")
    for i in range(num_samples):

        label = np.random.randint(0, 5)
        f0 = class_freqs[label]

        if label == 0:
            sig = sine_wave(f0, sr, dur)
        elif label == 1:
            sig = fm_signal(f0, sr, dur)
        elif label == 2:
            sig = am_signal(f0, sr, dur)
        elif label == 3:
            sig = pulse_signal(f0, sr, dur)
        else:
            sig = white_noise(sr, dur)

        fft = np.fft.fft(sig)
        mag = np.abs(fft[:num_features])

        # Normalize magnitude
        mag = mag / (np.max(mag) + 1e-12)

        X.append(mag)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    df = pd.DataFrame(X)
    df["label"] = y
    df.to_csv(out_csv, index=False)

    print(f"\nSaved dataset to {out_csv}")
    plot_examples(X, y, class_names, sr, dur, num_features)


def plot_examples(X, y, class_names, sr, dur, num_features):
    """Plot first example of each class."""
    N = int(sr * dur)
    freq = np.fft.fftfreq(N, 1/sr)[:num_features]

    plt.figure(figsize=(14, 10))
    classes = np.unique(y)

    for idx, c in enumerate(classes):
        i = np.where(y == c)[0][0]
        spec = X[i]

        plt.subplot(3, 2, idx+1)
        plt.plot(freq[:num_features//2], spec[:num_features//2])
        plt.title(f"{c}: {class_names[c]}")
        plt.xlabel("Hz")
        plt.ylabel("Normalized Mag")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig("data/example_spectra.png")
    plt.show()
    print("\nExample spectra saved to data/example_spectra.png")


# ------------------------------------------------------------

if __name__ == "__main__":
    create_dataset()
