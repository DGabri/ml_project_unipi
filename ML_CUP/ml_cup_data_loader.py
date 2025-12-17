from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy import loadtxt
import pandas as pd
import numpy as np
import os

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

    X = training_set.iloc[:, :-4]
    y = training_set.iloc[:, -4:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = 0.8)

    return X_train, X_test, y_train, y_test

def load_blind_set():
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

def scale_data(X_train, X_test, X_blind, y_train, y_test):
    
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    # scale inputs
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    X_blind_scaled = feature_scaler.transform(X_blind)
    
    # scale targets
    y_train_scaled = target_scaler.fit_transform(y_train)
    y_test_scaled  = target_scaler.transform(y_test)
    
    return (X_train_scaled, X_test_scaled, X_blind_scaled,
            y_train_scaled, y_test_scaled,
            feature_scaler, target_scaler)