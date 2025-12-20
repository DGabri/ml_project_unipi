import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from warnings import filterwarnings
from ml_cup_data_loader import (
    load_training_set,
    load_test_set,
    euclidean_distance_score,
    mee_scorer,
    write_blind_results
)

filterwarnings("ignore")

# load datasets
train_df = load_training_set()
blind_df = load_test_set()

X = train_df.iloc[:, :-4].values
y = train_df.iloc[:, -4:].values
X_blind = blind_df.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale features and target
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)
X_blind_scaled = feature_scaler.transform(X_blind)

target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train)

# grid search initialization
param_grid = {
    'estimator__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'estimator__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sag'],
}

# model and cross validation
ridge = Ridge()
multi_ridge = MultiOutputRegressor(ridge)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scorer_sklearn = make_scorer(mee_scorer, greater_is_better=True)

gs = GridSearchCV(multi_ridge, param_grid, scoring=scorer_sklearn, cv=kf, n_jobs=-1, verbose=1)
gs.fit(X_train_scaled, y_train_scaled)

print("best config:", gs.best_params_)

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_scaled, y_train_scaled, test_size=0.2, random_state=42
)

# retrain final model
best_model = gs.best_estimator_
best_model.fit(X_train_final, y_train_final)

y_train_pred = best_model.predict(X_train_final)
y_val_pred = best_model.predict(X_val)


train_loss = euclidean_distance_score(
    target_scaler.inverse_transform(y_train_final), 
    target_scaler.inverse_transform(y_train_pred)
)

val_loss = euclidean_distance_score(
    target_scaler.inverse_transform(y_val), 
    target_scaler.inverse_transform(y_val_pred)
)


y_pred = best_model.predict(X_test_scaled)
y_pred = target_scaler.inverse_transform(y_pred)

test_mee = euclidean_distance_score(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print("\nFinal results:\n")
print("Train Loss:", train_loss)
print("Val Loss:", val_loss)
print("Test MEE:", test_mee)
print("Test R²:", test_r2)