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