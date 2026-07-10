# Handwritten Digit Recognition Using Convolutional Neural Networks (CNN)

## Project Overview

This project presents a **Handwritten Digit Recognition System** developed using **Deep Learning** and **Convolutional Neural Networks (CNNs)**. The model is trained on the **MNIST dataset** to classify handwritten digits (0–9) with high accuracy. The project demonstrates the complete deep learning workflow, including data preprocessing, model development, training, evaluation, and performance analysis.

---

## Problem Statement

To design and implement a CNN-based image classification system capable of accurately recognizing handwritten digits from grayscale images and evaluating its performance using standard machine learning metrics.

---

## Objectives

* Develop a robust CNN architecture for handwritten digit recognition.
* Preprocess image data for efficient model training.
* Train and validate the model using the MNIST dataset.
* Evaluate model performance using Accuracy, Precision, and Confusion Matrix.
* Provide a scalable foundation for future OCR (Optical Character Recognition) applications.

---

## Dataset

**Dataset:** MNIST (Modified National Institute of Standards and Technology)

| Feature         | Value           |
| --------------- | --------------- |
| Training Images | 60,000          |
| Testing Images  | 10,000          |
| Image Size      | 28 × 28 pixels  |
| Classes         | 10 (Digits 0–9) |
| Image Format    | Grayscale       |

The dataset is loaded directly using TensorFlow:

```python
tf.keras.datasets.mnist.load_data()
```

---

## Technology Stack

* **Programming Language:** Python 3
* **Deep Learning Framework:** TensorFlow / Keras
* **Libraries:**

  * NumPy
  * Scikit-learn
  * Matplotlib
  * Seaborn

---

## Model Architecture

The CNN architecture consists of the following layers:

* Convolution Layer (32 Filters, ReLU)
* Batch Normalization
* Max Pooling
* Dropout (25%)
* Convolution Layer (64 Filters, ReLU)
* Batch Normalization
* Max Pooling
* Dropout (25%)
* Flatten Layer
* Dense Layer (128 Neurons, ReLU)
* Batch Normalization
* Dropout (50%)
* Output Layer (Softmax – 10 Classes)

**Optimizer:** Adam

**Loss Function:** Sparse Categorical Crossentropy

**Evaluation Metric:** Accuracy

---

## Project Workflow

1. Load the MNIST dataset.
2. Normalize image pixel values.
3. Reshape images for CNN input.
4. Build the CNN architecture.
5. Train the model on the training dataset.
6. Evaluate the model on the test dataset.
7. Generate performance metrics and confusion matrix.
8. Analyze digit-wise classification accuracy.

---

## Performance Evaluation

The model is evaluated using the following metrics:

* **Accuracy**
* **Macro Precision**
* **Confusion Matrix**
* **Digit-wise Classification Analysis**

These metrics provide a comprehensive assessment of the model's classification performance across all digit classes.

---

## Project Structure

```
Handwritten-Digit-Recognition/
│
├── handwritten_recognition.py
├── README.md
└── requirements.txt
```

---

## Installation

Install the required dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn
```

---

## Execution

Run the project using:

```bash
python handwritten_recognition.py
```

---

## Expected Output

The application generates:

* CNN Model Summary
* Training and Validation Performance
* Overall Test Accuracy
* Precision Score
* Confusion Matrix
* Digit-wise Classification Report

---

## Future Enhancements

* Handwritten alphabet recognition
* Word and sentence recognition
* CRNN (CNN + Bidirectional LSTM)
* CTC Loss for sequence prediction
* Real-time handwriting recognition
* OCR-based document digitization

---

## Applications

* Optical Character Recognition (OCR)
* Bank cheque digit recognition
* Postal code recognition
* Automated form processing
* Educational assessment systems
* Document digitization

---

## Author

**Lucky Rahangdale**
B.Tech in Artificial Intelligence

---

## Acknowledgements

* TensorFlow & Keras
* Scikit-learn
* NumPy
* MNIST Dataset
