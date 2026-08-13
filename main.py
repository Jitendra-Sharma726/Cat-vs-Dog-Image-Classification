import os
import pickle
import numpy as np
import cv2  # OpenCV for image processing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_preprocessed_data(filepath):
    """
    Load preprocessed train/test data and scaler from a pickle file.

    Args:
        filepath (str): Path to pickle file.

    Returns:
        tuple: (X_train, y_train, X_test, y_test, scaler)
    """
    with open(filepath, "rb") as f:
        X_train, y_train, X_test, y_test, scaler = pickle.load(f)
    return X_train, y_train, X_test, y_test, scaler


def train_model(X_train, y_train):
    """
    Train an MLPClassifier with one hidden layer.
    
    Args:
        X_train (np.ndarray): Training feature set.
        y_train (np.ndarray): Training labels.

    Returns:
        MLPClassifier: Trained model.
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=(8,),   # One hidden layer with 8 neurons
        activation="relu",
        solver="adam",
        max_iter=50,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        verbose=True
    )
    mlp.fit(X_train, y_train)
    return mlp


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and return accuracy and classification report.

    Args:
        model (MLPClassifier): Trained classifier.
        X_test (np.ndarray): Test feature set.
        y_test (np.ndarray): True test labels.

    Returns:
        tuple: (accuracy, classification_report_str)
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, report


def predict_image(model, scaler, image_path, image_size=(16, 16)):
    """
    Predict whether a given image is a Cat or Dog.

    Args:
        model (MLPClassifier): Trained model.
        scaler (StandardScaler): Scaler used for normalization.
        image_path (str): Path to the input image.
        image_size (tuple): Desired resize dimensions (width, height).

    Returns:
        str: "Cat" or "Dog" or error message.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return f"Error: Could not read image {image_path}"

        img_resized = cv2.resize(img, image_size)
        img_flattened = img_resized.flatten().reshape(1, -1)
        img_scaled = scaler.transform(img_flattened)

        pred = model.predict(img_scaled)[0]
        return "Dog" if pred == 1 else "Cat"
    except Exception as e:
        return f"Error processing image {image_path}: {e}"


if __name__ == "__main__":
    # Load preprocessed data
    X_train, y_train, X_test, y_test, scaler = load_preprocessed_data("scaled_data.pkl")
    print("Train Shape:", X_train.shape)
    print("Test Shape:", X_test.shape)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    acc, report = evaluate_model(model, X_test, y_test)
    print("Accuracy:", acc)
    print("Classification Report:\n", report)

    # Save trained model
    with open("cat_dog_model.pkl", "wb") as f:
        pickle.dump((model, scaler), f)
    print("Model saved as cat_dog_model.pkl")

    # Predict on sample images
    test_images = ["dog.jpg", "cat.jpg"]  # Replace with actual paths
    for img_path in test_images:
        result = predict_image(model, scaler, img_path)
        print(f"Prediction for {img_path}: {result}")
