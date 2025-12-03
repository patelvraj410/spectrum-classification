"""
Data processing module for RF spectrum classification.
Handles loading, preprocessing, and preparing RF spectrum data.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataProcessor:
    def __init__(self, test_size=0.2, random_state=42):
        """Initialize the data processor.
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def load_data(self, filepath):
        """Load RF spectrum data from a file.
        
        Args:
            filepath (str): Path to the data file
            
        Returns:
            tuple: (features, labels) as numpy arrays
        """
        # TODO: Implement data loading logic based on your data format
        # This is a placeholder - update with actual data loading code
        data = pd.read_csv(filepath)
        X = data.drop('label', axis=1).values
        y = data['label'].values
        return X, y
    
    def preprocess_data(self, X, y):
        """Preprocess the RF spectrum data.
        
        Args:
            X (numpy.ndarray): Input features
            y (numpy.ndarray): Target labels
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y_encoded
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
