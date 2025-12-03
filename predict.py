import joblib
import numpy as np
import pandas as pd

import argparse
import os

def predict_new_sample(model_path, X_new):
    """
    Make predictions using the trained model.
    
    Args:
        model_path (str): Path to the saved model
        X_new (np.array): New data samples to predict (shape: n_samples, n_features)
        
    Returns:
        np.array: Predicted class labels
    """
    # Load the model
    model = joblib.load(model_path)
    
    # Make predictions
    predictions = model.predict(X_new)
    
    # Map numeric predictions to class names
    class_names = {
        0: 'Sine Wave',
        1: 'FM Signal',
        2: 'AM Signal',
        3: 'Pulse Signal',
        4: 'Noise'
    }
    
    # Convert numeric predictions to class names
    predicted_classes = [class_names[pred] for pred in predictions]
    
    return predicted_classes

def main():
    parser = argparse.ArgumentParser(description="Predict RF spectrum classes using a saved Random Forest model.")
    parser.add_argument("--csv", type=str, default="data/predict_input.csv", help="Path to CSV file with samples (label column optional and ignored).")
    parser.add_argument("--model", type=str, default="models/rf_model.pkl", help="Path to saved joblib model.")
    parser.add_argument("--out", type=str, default="predictions/predictions.csv", help="Where to save the predictions CSV.")
    parser.add_argument("--head", type=int, default=10, help="Print first N rows of predictions.")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: model not found at {args.model}. Train it with: python train.py")
        return
    if not os.path.exists(args.csv):
        print(f"Error: input CSV not found at {args.csv}. Create one with: python make_predict_file.py")
        return

    # Load model and input data
    model = joblib.load(args.model)
    df = pd.read_csv(args.csv)
    has_label = 'label' in df.columns
    X = df.drop('label', axis=1).values if has_label else df.values
    y_true = df['label'].values if has_label else None

    # Predict
    preds_idx = model.predict(X)
    class_names = ['Sine Wave', 'FM Signal', 'AM Signal', 'Pulse Signal', 'Noise']
    preds_name = [class_names[i] if 0 <= i < len(class_names) else str(i) for i in preds_idx]

    # Save predictions
    out_df = df.copy()
    out_df['predicted'] = preds_idx
    out_df['predicted_name'] = preds_name
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[INFO] Saved predictions to {os.path.abspath(args.out)}")

    # Print head
    n = min(args.head, len(out_df))
    print(f"\nPredictions (first {n} rows):")
    header = f"{'Idx':<6} {'Pred':<5} {'Name':<15}"
    if y_true is not None:
        header += " True"
    print(header)
    for i in range(n):
        row = f"{i:<6} {preds_idx[i]:<5} {preds_name[i]:<15}"
        if y_true is not None:
            true_name = class_names[y_true[i]] if 0 <= y_true[i] < len(class_names) else str(y_true[i])
            row += f" {true_name}"
        print(row)

if __name__ == "__main__":
    main()
