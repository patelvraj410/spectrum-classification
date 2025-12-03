"""
Example usage of the RF Spectrum Classification project.

This script demonstrates how to use the project components to train and evaluate
a model for RF spectrum classification using the synthetic dataset.
"""
import os
import numpy as np
import pandas as pd
from src.data_processing import DataProcessor
from src.models import SpectrumClassifier
from src.visualization import plot_training_history, plot_confusion_matrix

def load_synthetic_dataset(csv_path='data/rf_spectrum_dataset.csv'):
    """Load the synthetic RF spectrum dataset."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please run dataset_generator.py first.")
    
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Separate features and labels
    X = df.drop('label', axis=1).values
    y = df['label'].values
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} features each")
    print(f"Number of classes: {len(np.unique(y))}")
    
    return X, y

def main():
    # Initialize paths
    data_dir = 'data'
    model_dir = 'models'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # Load the dataset
        X, y = load_synthetic_dataset()
        
        # Initialize data processor
        print("Initializing data processor...")
        processor = DataProcessor(test_size=0.2, random_state=42)
        
        # Split data into train and test sets
        print("Preprocessing data...")
        X_train, X_test, y_train, y_test = processor.preprocess_data(X, y)
        
        # Reshape data for CNN (add channel dimension)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Initialize and train model
        print("Initializing model...")
        input_shape = (X_train.shape[1], 1)  # (timesteps, features)
        num_classes = len(np.unique(y))
        model = SpectrumClassifier(input_shape=input_shape, num_classes=num_classes)
        
        # Split training data into training and validation sets
        val_size = int(0.2 * len(X_train))
        X_train, X_val = X_train[:-val_size], X_train[-val_size:]
        y_train, y_val = y_train[:-val_size], y_train[-val_size:]
        
        print("\nModel architecture:")
        model.model.summary()
        
        print("\nTraining model...")
        history = model.train(
            X_train, y_train, 
            X_val, y_val,
            epochs=50,
            batch_size=64,  # Increased batch size for better stability
            model_dir=model_dir
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
        
        # Make predictions
        y_pred = np.argmax(model.predict(X_test), axis=1)
        
        # Plot training history
        plot_training_history(history)
        
        # Plot confusion matrix
        class_names = [
            'Sine Wave', 'FM Signal', 'AM Signal', 
            'Pulse Signal', 'Noise'
        ]
        plot_confusion_matrix(y_test, y_pred, classes=class_names)
        
        # Save the final model
        model_path = os.path.join(model_dir, 'rf_spectrum_classifier.h5')
        model.save(model_path)
        print(f"\nModel saved to {os.path.abspath(model_path)}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nMake sure you've run dataset_generator.py first to create the synthetic dataset.")
        print("You can run it with: python dataset_generator.py")

if __name__ == "__main__":
    main()
