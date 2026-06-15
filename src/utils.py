import os
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_landmark_csv(label, landmark_vector, dataset_dir="dataset"):
    """
    Save or append a numpy array row to dataset/<label>/landmarks.npy
    landmark_vector: 1D numpy array (126,)
    """
    folder = os.path.join(dataset_dir, label)
    ensure_dir(folder)
    file_path = os.path.join(folder, "landmarks.npy")
    if os.path.exists(file_path):
        arr = np.load(file_path)
        arr = np.vstack([arr, landmark_vector])
    else:
        arr = np.array([landmark_vector])
    np.save(file_path, arr)


def load_dataset(dataset_dir="dataset"):
    """
    Loads all dataset/<label>/landmarks.npy files and returns
    X (n_samples, n_features), y (n_samples,), labels (list)
    """
    X = []
    y = []
    labels = sorted(
        [
            d
            for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
        ]
    )
    for lbl in labels:
        file = os.path.join(dataset_dir, lbl, "landmarks.npy")
        if not os.path.exists(file):
            continue
        data = np.load(file)
        for row in data:
            X.append(row)
            y.append(lbl)
    if len(X) == 0:
        return np.array([]), np.array([]), labels
    return np.array(X), np.array(y), labels
