import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv("data/rf_spectrum_dataset.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

print("[INFO] Dataset shape:", X.shape)

# ---------------------------------------------------------
# Train-test split
# ---------------------------------------------------------
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# Random Forest Model
# ---------------------------------------------------------
print("Initializing Random Forest model...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

print("\n[INFO] Training model...")
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "rf_model.pkl")
joblib.dump(model, model_path)
print(f"[INFO] Model saved to {os.path.abspath(model_path)}")

# ---------------------------------------------------------
# Evaluate
# ---------------------------------------------------------
print("\n[INFO] Evaluating model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\nAccuracy: {:.2f}%".format(acc * 100))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Sine Wave', 'FM Signal', 'AM Signal', 'Pulse Signal', 'Noise']))

# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------
print("\n[INFO] Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", 
            xticklabels=['Sine', 'FM', 'AM', 'Pulse', 'Noise'],
            yticklabels=['Sine', 'FM', 'AM', 'Pulse', 'Noise'])
plt.title("RF Spectrum Classification - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

os.makedirs("plots", exist_ok=True)
conf_matrix_path = os.path.join("plots", "confusion_matrix.png")
plt.savefig(conf_matrix_path)
print(f"[INFO] Confusion matrix saved to {os.path.abspath(conf_matrix_path)}")

print("\n[INFO] Training and evaluation complete!")
