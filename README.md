
# RF Spectrum Classification using Machine Learning & Deep Learning

This project implements a complete RF signal classification pipeline using:

- Synthetic RF signal generation (SciPy)
- FFT-based feature extraction (1024 bins)
- Random Forest baseline classifier
- 1D CNN deep learning classifier (TensorFlow/Keras)
- Visualization of spectra, confusion matrices & feature importance

It achieves **100% accuracy** with the CNN model.

---

## Project Overview

RF signal classification is critical in wireless communication, spectrum monitoring, cognitive radio, SDR, and signal intelligence.

Signal classes:

0 - Sine Wave
1 - Frequency Modulated (FM)
2 - Amplitude Modulated (AM)
3 - Pulse Signal
4 - Noise

---

## Folder Structure

project-root/
│
├── dataset_generator.py
├── example_spectra.png
├── feature_importances.png
├── confusion_matrix.png
├── rf_spectrum_dataset.csv
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── models.py
│   ├── train_cnn.py
│   ├── visualization.py
│
├── models/
│   ├── rf_cnn_model.keras
│
├── predictions/
│   ├── predictions.csv
│   ├── predict_template.csv
│   ├── predict_input.csv

---

## Dataset Generation

Run:
python dataset_generator.py

---

## Machine Learning (Random Forest)

- Uses FFT vectors
- Generates feature_importances.png
- Generates confusion_matrix.png

---

## Deep Learning Model (1D CNN)

Run:
python src/train_cnn.py

Achieves **100% accuracy**.

---

## Prediction

Use template:
predictions/predict_template.csv

Run:
python predict.py --input predictions/predict_input.csv

---

## License
MIT License
