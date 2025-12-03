import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_processing import DataProcessor
from src.models import SpectrumClassifier
from src.visualization import plot_training_history, plot_confusion_matrix


def main():
    data_csv = os.path.join('data', 'rf_spectrum_dataset.csv')
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"Dataset not found at {data_csv}. Run dataset_generator.py first.")

    # Load dataset
    print(f"[INFO] Loading dataset from {data_csv} ...")
    df = pd.read_csv(data_csv)
    X = df.drop('label', axis=1).values
    y = df['label'].values
    num_classes = len(np.unique(y))
    print(f"[INFO] Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {num_classes}")

    # Preprocess (scale + split)
    print("[INFO] Preprocessing (scaling + split) ...")
    processor = DataProcessor(test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = processor.preprocess_data(X, y)

    # Reshape for 1D CNN: (samples, timesteps, channels)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build model
    input_shape = (X_train.shape[1], 1)
    print(f"[INFO] Building model with input_shape={input_shape} and {num_classes} classes ...")
    model = SpectrumClassifier(input_shape=input_shape, num_classes=num_classes)

    # Make a validation split from the training set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,  # Ensure balanced classes in both splits
        shuffle=True
    )
    print(f"[INFO] Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Train model
    print("[INFO] Training model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=50,  # You can adjust the number of epochs
        batch_size=32,  # You can adjust the batch size
        model_dir=model_dir
    )

    # Evaluate
    print("[INFO] Evaluating on test set ...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"[RESULT] Test Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")

    # Save the final model
    model_path = os.path.join(model_dir, 'rf_cnn_model.keras')
    model.save(model_path)
    print(f"[INFO] Model saved to {model_path}")

    # Predict and visualize
    print("[INFO] Generating predictions and plots ...")
    y_pred = np.argmax(model.predict(X_test), axis=1)

    try:
        plot_training_history(history)
        class_names = ['Sine Wave', 'FM Signal', 'AM Signal', 'Pulse Signal', 'Noise']
        plot_confusion_matrix(y_test, y_pred, classes=class_names)
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")

    # Save final model
    final_path = os.path.join(model_dir, 'rf_cnn_model.keras')
    model.save(final_path)
    print(f"[INFO] Saved final model to {os.path.abspath(final_path)}")


if __name__ == '__main__':
    main()
