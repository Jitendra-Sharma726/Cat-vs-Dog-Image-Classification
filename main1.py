import os
import numpy as np
import cv2  # OpenCV for image processing

def load_images_from_folder(folder, label, image_size = (16, 16)):
    """
    Load images from a given folder, resize, flatten, and assign a label.

    Args:
        folder (str): Path to the folder containing images.
        label (int): Label assigned to all images in this folder.
        image_size (tuple): Desired image size (width, height).

    Returns:
        tuple: images, labels
    """
    images, labels = [], []

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        img = cv2.imread(file_path)
        if img is not None:
            img_resized = cv2.resize(img, image_size)
            images.append(img_resized.flatten())
            labels.append(label)

    return images, labels


def load_dataset(train_dir, test_dir):
    """
    Load training and testing datasets from given directories.

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


if __name__ == "__main__":
    train_dir = "train"
    test_dir = "test"

    print("Loading dataset...")
    X_train, y_train, X_test, y_test = load_dataset(train_dir, test_dir)

    print("Dataset ready!")
    # Train set info
    print(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
    print(f" - Cats: {np.sum(y_train == 0)}")
    print(f" - Dogs: {np.sum(y_train == 1)}\n")

    # Test set info
    print(f"Test set shape:  X={X_test.shape}, y={y_test.shape}")
    print(f" - Cats: {np.sum(y_test == 0)}")
    print(f" - Dogs: {np.sum(y_test == 1)}\n")
