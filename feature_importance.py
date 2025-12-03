import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importances():
    """Plot feature importances from the trained Random Forest model."""
    # Load the trained model
    print("Loading trained model...")
    model = joblib.load("models/rf_model.pkl")
    
    # Load the dataset to get feature names
    import pandas as pd
    df = pd.read_csv("data/rf_spectrum_dataset.csv")
    X = df.drop("label", axis=1)
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot the feature importances
    plt.figure(figsize=(15, 8))
    plt.title("Top 20 Most Important Features")
    plt.bar(range(20), importances[indices][:20], align="center")
    plt.xticks(range(20), [f"Freq {i}" for i in indices[:20]], rotation=45, ha='right')
    plt.xlim([-1, 20])
    plt.ylabel("Importance")
    plt.tight_layout()
    
    # Save the plot
    import os
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/feature_importances.png")
    print("Feature importances plot saved to plots/feature_importances.png")
    
    # Show the plot
    plt.show()
    
    # Print top 10 most important features
    print("\nTop 10 Most Important Features:")
    for i in range(10):
        print(f"{i+1}. Feature {indices[i]}: {importances[indices[i]]:.6f}")

if __name__ == "__main__":
    plot_feature_importances()
