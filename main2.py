import os
import numpy as np
import cv2  # OpenCV for image reading and resizing
import pickle
from sklearn.preprocessing import StandardScaler


def load_images_from_folder(folder, label, image_size=(16, 16)):
    """
    Load images from a given folder, resize, flatten, and assign labels.

    Args:
        folder (str): Path to the folder containing images.
        label (int): Label assigned to all images in this folder.
        image_size (tuple): Desired image size (width, height).

    Returns:
        tuple: (images, labels) where
            images (list): Flattened image arrays.
            labels (list): Corresponding labels.
    """
    images, labels = [], []

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        img = cv2.imread(file_path)  # Read the image
        if img is not None:  # Only process valid images
            img_resized = cv2.resize(img, image_size)  # Resize
            images.append(img_resized.flatten())       # Flatten to 1D
            labels.append(label)

    return images, labels


def load_dataset(train_dir, test_dir):
    """
    Load training and testing datasets from given directories.
    Directory structure:
        train/
            cats/
            dogs/
        test/
            cats/
            dogs/

    Args:
        train_dir (str): Path to training dataset directory.
        test_dir (str): Path to testing dataset directory.

    Returns:
        tuple: (X_train, y_train, X_test, y_test) as numpy arrays.
    """
    # Load training data
    cat_train, cat_train_labels = load_images_from_folder(os.path.join(train_dir, "cats"), 0)
    dog_train, dog_train_labels = load_images_from_folder(os.path.join(train_dir, "dogs"), 1)

    # Load testing data
    cat_test, cat_test_labels = load_images_from_folder(os.path.join(test_dir, "cats"), 0)
    dog_test, dog_test_labels = load_images_from_folder(os.path.join(test_dir, "dogs"), 1)

    # Combine training
    X_train = np.array(cat_train + dog_train)
    y_train = np.array(cat_train_labels + dog_train_labels)

    # Combine testing
    X_test = np.array(cat_test + dog_test)
    y_test = np.array(cat_test_labels + dog_test_labels)

    return X_train, y_train, X_test, y_test


def scale_features(X_train, X_test):
    """
    Scale dataset features using StandardScaler.

    Args:
        X_train (np.ndarray): Training feature set.
        X_test (np.ndarray): Testing feature set.

    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    # Paths to dataset
    train_dir = "train"
    test_dir = "test"

    # Load dataset
    X_train, y_train, X_test, y_test = load_dataset(train_dir, test_dir)
    print("Original Train Shape:", X_train.shape)
    print("Original Test Shape:", X_test.shape)

    # Scale dataset
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    print("Scaled Train Shape:", X_train_scaled.shape)
    print("Scaled Test Shape:", X_test_scaled.shape)
    
    # Show before & after scaling (first sample, first 3 features)
    print("\nSample Features (First Image):")
    print(" - Before Scaling:", X_train[0][:3])
    print(" - After Scaling :", X_train_scaled[0][:3])

    # Save scaled data + scaler for later use
    with open("scaled_data.pkl", "wb") as f:
        pickle.dump((X_train_scaled, y_train, X_test_scaled, y_test, scaler), f)
