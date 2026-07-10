import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers, models

class HandwrittenRecognitionPipeline:
    """
    A Deep Learning Pipeline for Handwritten Character Recognition
    utilizing Convolutional Neural Networks (CNN).
    """
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_cnn_architecture()

    def _build_cnn_architecture(self):
        """
        Defines a robust Convolutional Neural Network structure optimal for 
        handling pixel feature extractions from handwritten characters.
        """
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
        
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def train_pipeline(self, X_train, y_train, X_val, y_val, epochs=5, batch_size=64):
        """Trains the CNN architecture on the character data."""
        print("\n" + "="*60 + "\n[+] Commencing Model Training Process\n" + "="*60)
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val)
        )
        return history

    def evaluate_performance(self, X_test, y_test):
        """Generates Accuracy, Precision, and Detailed Confusion Matrix."""
        print("\n" + "="*60 + "\n[+] Generating System Performance Diagnostics\n" + "="*60)
        
        # Predict class probabilities and derive classes
        probabilities = self.model.predict(X_test)
        predictions = np.argmax(probabilities, axis=1)
        
        # Calculate scores
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='macro', zero_division=0)
        matrix = confusion_matrix(y_test, predictions)
        
        # Print report card
        print(f"[*] Accuracy Score  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"[*] Precision Score : {precision:.4f} ({precision * 100:.2f}%)")
        print("\n[*] Confusion Matrix Array Distribution:")
        print(matrix)
        
        return predictions

# DATA ENGINE & LIFECYCLE MANAGEMENT

if __name__ == "__main__":
    # 1. Fetching Dataset 
    print("[+] Fetching and preprocessing character pixel vectors...")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 2. Image Processing
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    # 3. Instantiate Engine Pipeline
    num_classes = len(np.unique(y_train))
    pipeline = HandwrittenRecognitionPipeline(input_shape=(28, 28, 1), num_classes=num_classes)
    
    
    pipeline.model.summary()

    # 4. Train the Neural Network
    pipeline.train_pipeline(X_train, y_train, X_test, y_test, epochs=3, batch_size=128)

    # 5. Evaluate Accuracy, Precision, and Confusion Matrix
    predictions = pipeline.evaluate_performance(X_test, y_test)

    # 6. Architectural Expansion Overview: Transitioning to Sequence Modeling (CRNN)
    print("\n" + "="*70 + "\nPROPOSED ROUTEMAP FOR FULL SENETENCE/WORD SEQUENCE MODELING (CRNN)\n" + "="*70)
    print(
        "To extend this system beyond individual characters to full words or sentences:\n\n"
        "1. Remove Dense layers from the CNN above to preserve the spatial feature map.\n"
        "2. Feed the sequential text features into a Recurrent Neural Network (Bidirectional LSTM).\n"
        "3. Apply a Connectionist Temporal Classification (CTC) Loss layer at the top.\n"
        "   This allows the network to read continuous strings of text without requiring pre-sliced character images."
    )
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import seaborn as sns # Optional: for plotting a pretty heatmap

# 1. Get raw model probability predictions on your test dataset
probabilities = pipeline.model.predict(X_test)

# 2. Convert raw probabilities to hard index labels (0 to 9) using argmax
predictions = np.argmax(probabilities, axis=1)

# 3. Compute metric indices
accuracy = accuracy_score(y_test, predictions)

# 'macro' average calculates metrics independently for each digit class and finds their unweighted mean
precision = precision_score(y_test, predictions, average='macro', zero_division=0)

# 4. Generate the raw array matrix
matrix = confusion_matrix(y_test, predictions)


# SYSTEM PERFORMANCE METRICS DISPLAY PANEL

print("\n" + "="*50)
print("     CNN CHARACTER RECOGNITION REPORT CARD")
print("="*50)
print(f"[+] Total Evaluation Samples Processed : {len(y_test)}")
print(f"[+] Overall Model Accuracy             : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"[+] Multi-Class Macro Precision Score  : {precision:.4f} ({precision * 100:.2f}%)")
print("\n[+] Raw Confusion Matrix Spatial Array:")
print(matrix)

print("\n" + "-"*50)
print("  DIGIT-BY-DIGIT MISCLASSIFICATION ANALYSIS")
print("-"*50)
for i in range(num_classes):
    true_positive = matrix[i, i]
    total_actual = np.sum(matrix[i, :])
    total_predicted = np.sum(matrix[:, i])
    
    class_acc = true_positive / total_actual if total_actual > 0 else 0
    print(f"Digit [{i}] -> Correctly Classified: {true_positive}/{total_actual} ({class_acc*100:.1f}%)")
