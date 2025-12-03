"""
Visualization utilities for RF spectrum classification.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    roc_curve, 
    auc,
    precision_recall_curve,
    average_precision_score
)
import seaborn as sns


def plot_spectrum(spectrum, fs, title='RF Spectrum', xlabel='Frequency (Hz)', ylabel='Amplitude'):
    """Plot a single RF spectrum.
    
    Args:
        spectrum (numpy.ndarray): Spectrum data
        fs (float): Sampling frequency
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
    """
    plt.figure(figsize=(10, 5))
    freqs = np.fft.fftfreq(len(spectrum), 1/fs)
    plt.plot(freqs[:len(freqs)//2], np.abs(spectrum[:len(spectrum)//2]))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix'):
    """Plot a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        title: Plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """Plot training and validation accuracy/loss over epochs.
    
    Args:
        history: Keras History object returned from model.fit()
    """
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_scores, num_classes, class_names=None):
    """Plot ROC curves for each class.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_scores: Predicted probabilities for each class
        num_classes: Number of classes
        class_names: List of class names
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot all ROC curves
    plt.figure(figsize=(8, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(importances, feature_names, top_n=20):
    """Plot feature importance.
    
    Args:
        importances: Feature importance scores
        feature_names: List of feature names
        top_n: Number of top features to show
    """
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Rearrange feature names so they match the sorted feature importances
    names = [feature_names[i] for i in indices]
    
    # Limit to top_n features
    if top_n is not None:
        indices = indices[:top_n]
        names = names[:top_n]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), names, rotation=90)
    plt.xlim([-1, len(indices)])
    plt.tight_layout()
    plt.show()
