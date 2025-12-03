"""
Model definitions for RF spectrum classification.
"""
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os

class SpectrumClassifier:
    """A CNN-based classifier for RF spectrum classification."""
    
    def __init__(self, input_shape, num_classes):
        """Initialize the model.
        
        Args:
            input_shape (tuple): Shape of input data (timesteps, features)
            num_classes (int): Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self):
        """Build the CNN model architecture."""
        model = Sequential([
            # First Conv1D layer
            Conv1D(filters=64, kernel_size=3, activation='relu', 
                  input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            # Second Conv1D layer
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            # Third Conv1D layer
            Conv1D(filters=256, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            # Dense layers
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=100, batch_size=32, model_dir='models'):
        """Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            model_dir: Directory to save model checkpoints
            
        Returns:
            History object containing training history
        """
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Define callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(model_dir, 'best_model.h5'),
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            tuple: (loss, accuracy)
        """
        return self.model.evaluate(X_test, y_test, verbose=0)
    
    def predict(self, X):
        """Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            numpy.ndarray: Predicted class probabilities
        """
        return self.model.predict(X, verbose=0)
    
    def save(self, filepath):
        """Save the model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load a saved model from a file.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            SpectrumClassifier: Loaded model instance
        """
        model = load_model(filepath)
        # Create a new instance and replace its model
        input_shape = model.layers[0].input_shape[1:]
        num_classes = model.layers[-1].output_shape[-1]
        instance = cls(input_shape, num_classes)
        instance.model = model
        return instance
