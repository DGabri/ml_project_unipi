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

def mee(y_true, y_pred):
    return np.mean(np.linalg.norm(y_true - y_pred, axis=1))

def mee_scorer(y_true, y_pred):
    return -mee(y_true, y_pred)
    
def euclidean_distance_loss(y_true, y_pred):
    return np.sqrt(np.sum(np.square(y_pred - y_true), axis=-1))

def euclidean_distance_score(y_true, y_pred):
    return np.mean(euclidean_distance_loss(y_true, y_pred))

def write_blind_results(model_name, y_pred):
    team_name = "GabriMassi"
    date_str = "07/01/2026"
    file_path = os.path.join(BASE_DIR, f"{team_name}_ML-CUP25-TS.csv")
    
    with open(file_path, "w") as f:
        f.write("# Gabriele Deri, Massimo Parlanti\n")
        f.write(f"# {team_name}\n")
        f.write("# ML-CUP25\n")  
        f.write(f"# {date_str}\n")  
        
        for pred_id, p in enumerate(y_pred, start=1):
            f.write(f"{pred_id},{float(p[0])},{float(p[1])},{float(p[2])},{float(p[3])}\n")

