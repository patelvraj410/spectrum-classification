import os
import pandas as pd

def main():
    src = "data/rf_spectrum_dataset.csv"
    dst = "data/predict_input.csv"
    tmpl = "data/predict_template.csv"

    if not os.path.exists(src):
        raise FileNotFoundError(f"Source dataset not found at {src}. Run dataset_generator.py first.")

    os.makedirs("data", exist_ok=True)

    # Load full dataset and drop label to create input-only file
    df = pd.read_csv(src)
    X = df.drop("label", axis=1)

    # Save a smaller sample for quick prediction, and a full version if desired
    X_sample = X.sample(n=min(50, len(X)), random_state=42)
    X_sample.to_csv(dst, index=False)
    print(f"Wrote sample prediction input to: {os.path.abspath(dst)}  ({len(X_sample)} rows)")

    # Also write a template with a single zero row to show exact format
    zeros = pd.DataFrame([ {col: 0.0 for col in X.columns} ])
    zeros.to_csv(tmpl, index=False)
    print(f"Wrote template to: {os.path.abspath(tmpl)}  (1 zeroed row)")

if __name__ == "__main__":
    main()
