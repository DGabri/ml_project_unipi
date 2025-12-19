from numpy import loadtxt
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_training_set():
    # laod 16 columns from csv with delimiter ','
    filename = BASE_DIR + "/dataset/ML-CUP25-TR.csv"
    training_rows = loadtxt(filename, delimiter=',', usecols=range(1, 17), dtype=np.float32)

    # get number of input columns
    num_inputs = training_rows.shape[1] - 4

    # prepare column names
    column_names = [f'INPUT_{i+1}' for i in range(num_inputs)] + \
                ['TARGET_1', 'TARGET_2', 'TARGET_3', 'TARGET_4']

    # convert to dataframe
    training_set = pd.DataFrame(training_rows, columns=column_names)

    return training_set

def load_test_set():
    # laod 16 columns from csv with delimiter ','
    filename = BASE_DIR + "/dataset/ML-CUP25-TS.csv"
    test_rows = loadtxt(filename, delimiter=',', usecols=range(1, 13), dtype=np.float32)

    # get number of input columns
    num_inputs = test_rows.shape[1]

    # prepare column names
    column_names = [f'INPUT_{i+1}' for i in range(num_inputs)]

    # convert to dataframe
    test_set = pd.DataFrame(test_rows, columns=column_names)

    return test_set

# -------------------------------
# Metric functions
# -------------------------------
def mee(y_true, y_pred):
    """Mean Euclidean Error (MEE)"""
    return np.mean(np.linalg.norm(y_true - y_pred, axis=1))

def mee_scorer(y_true, y_pred):
    """Scorer for GridSearchCV (negative because sklearn minimizes)"""
    return -mee(y_true, y_pred)

# -------------------------------
# Scaler utility
# -------------------------------
def scale_features_targets(X_train, X_val, X_test, X_blind, y_train=None):
    """
    Fit MinMaxScaler on X_train (and optionally y_train), 
    transform all datasets.
    """
    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled   = feature_scaler.transform(X_val)
    X_test_scaled  = feature_scaler.transform(X_test)
    X_blind_scaled = feature_scaler.transform(X_blind)

    if y_train is not None:
        target_scaler = MinMaxScaler()
        y_train_scaled = target_scaler.fit_transform(y_train)
        return X_train_scaled, X_val_scaled, X_test_scaled, X_blind_scaled, y_train_scaled, target_scaler
    else:
        return X_train_scaled, X_val_scaled, X_test_scaled, X_blind_scaled
    
def euclidean_distance_loss(y_true, y_pred):
    """
    Compute the Euclidean distance loss (element-wise).
    
    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
    
    Returns:
        np.ndarray: The Euclidean distance for each sample.
    """
    return np.sqrt(np.sum(np.square(y_pred - y_true), axis=-1))

def euclidean_distance_score(y_true, y_pred):
    """
    Compute the mean Euclidean distance between predictions and true values.
    
    Args:
        y_true (np.ndarray): Ground truth values.
        y_pred (np.ndarray): Predicted values.
    
    Returns:
        float: The mean Euclidean distance.
    """
    return np.mean(euclidean_distance_loss(y_true, y_pred))

def write_blind_results(model_name, y_pred):
    """
    Save predicted results in a CSV file for the blind test dataset.
    
    Args:
        y_pred (np.ndarray): The predictions to save.
    """
    
    # Create directory if it doesn't exist
    dir_path = os.path.join(BASE_DIR, model_name)
    os.makedirs(dir_path, exist_ok=True)

    file_path = os.path.join(dir_path, "ML-CUP25-TS.csv")
    with open(file_path, "w") as f:
        f.write("# Gabriele Deri \t Massimo Parlanti\n")

        for pred_id, p in enumerate(y_pred, start=1):
            f.write(f"{pred_id},{p[0]},{p[1]},{p[2]},{p[3]}\n")  # 4 target!

    print(f"\n✓ Saved {len(y_pred)} predictions to {file_path}")