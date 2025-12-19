import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.svm import SVR
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

# ======================================================
# Load datasets
# ======================================================
train_df = load_training_set()
blind_df = load_test_set()

X = train_df.iloc[:, :-4].values
y = train_df.iloc[:, -4:].values
X_blind = blind_df.values

# ======================================================
# Split: TR / TS (come il primo codice)
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======================================================
# Scale features and targets
# ======================================================
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_test_scaled = feature_scaler.transform(X_test)
X_blind_scaled = feature_scaler.transform(X_blind)

target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train)

# ======================================================
# Grid Search parameters (IDENTICI al primo codice)
# ======================================================
param_grid = { 
     'estimator__kernel': ['rbf'],
    'estimator__C': [1, 10, 40, 50, 100],
    'estimator__gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'estimator__epsilon': [0.01, 0.05, 0.1, 0.2]
}


# ======================================================
# Model and CV (10-fold come il primo)
# ======================================================
svr = SVR()
multi_svr = MultiOutputRegressor(svr)
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scorer_sklearn = make_scorer(mee_scorer, greater_is_better=True)

print("Starting Grid Search...")

start_time = time.time()
gs = GridSearchCV(multi_svr, param_grid, scoring=scorer_sklearn, cv=kf, n_jobs=-1, verbose=1)
gs.fit(X_train_scaled, y_train_scaled)
print(f"\nGrid Search completed in {time.time()-start_time:.4f} seconds.\n")

# Best parameters
print("Best parameters:", gs.best_params_)

# ======================================================
# Split train in train/val DOPO grid search (come il primo)
# ======================================================
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train_scaled, y_train_scaled, test_size=0.2, random_state=42
)

# ======================================================
# Train final model on training set
# ======================================================
best_model = gs.best_estimator_
best_model.fit(X_train_final, y_train_final)

# Predizioni su train e validation (SULLA SCALA SCALATA)
y_train_pred = best_model.predict(X_train_final)
y_val_pred = best_model.predict(X_val)

# Loss su scala scalata (come il primo codice)
train_loss = euclidean_distance_score(
    target_scaler.inverse_transform(y_train_final), 
    target_scaler.inverse_transform(y_train_pred)
)
val_loss = euclidean_distance_score(
    target_scaler.inverse_transform(y_val), 
    target_scaler.inverse_transform(y_val_pred)
)

print("Train Loss:", train_loss)
print("Val Loss:", val_loss)

# ======================================================
# Final evaluation on test set (original scale)
# ======================================================
y_pred = best_model.predict(X_test_scaled)
y_pred = target_scaler.inverse_transform(y_pred)
test_loss = euclidean_distance_score(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

print("---- Final Results (original scale data) ----")
print("Test Loss:", test_loss)
print("Test R²:", test_r2)

# ======================================================
# Blind test predictions
# ======================================================
y_blind_pred = best_model.predict(X_blind_scaled)
y_blind_pred = target_scaler.inverse_transform(y_blind_pred)
write_blind_results("SklearnSVM", y_blind_pred)


# Starting Grid Search...
# Fitting 10 folds for each of 100 candidates, totalling 1000 fits

# Grid Search completed in 6.3804 seconds.

# Best parameters: {'estimator__C': 10, 'estimator__epsilon': 0.2, 'estimator__gamma': 1, 'estimator__kernel': 'rbf'}
# Train Loss: 7.506081437724274
# Val Loss: 18.33804593526876
# ---- Final Results (original scale data) ----
# Test Loss: 18.25680845461501
# Test R²: 0.6019143115733867
