import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_monk_data(dataset_id):
    # load training and test set
    
    # columns: NaN target a1 a2 a3 a4 a5 a6 ID
    # usecols=range(1,8) to skip first empty column
    train = pd.read_csv(f"dataset/monks-{dataset_id}.train", sep=' ', header=None, usecols=range(1, 8))
    test = pd.read_csv(f"dataset/monks-{dataset_id}.test", sep=' ', header=None, usecols=range(1, 8))

    # extract columns 1 to 7 which are the features
    X_train = train.iloc[:, 1:]
    # extract column 0 which is the target
    y_train = train.iloc[:, 0]

    X_test = test.iloc[:, 1:]
    y_test = test.iloc[:, 0]

    # one-hot encoder fit categorical features -> binary columns
    # sparse_output=False returns a dense array instead of sparse matrix to save space
    encoder = OneHotEncoder(sparse_output = False)

    # appluy input encoding to trainign and test set
    X_train = pd.DataFrame(encoder.fit_transform(X_train)).astype(int)
    X_test = pd.DataFrame(encoder.transform(X_test)).astype(int)

    return (
        X_train,
        y_train,
        X_test,
        y_test    
    )