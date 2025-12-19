import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
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
# Split: TR / VL / TS
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# ======================================================
# Scale features and targets
# ======================================================
feature_scaler = StandardScaler()
X_train_scaled = feature_scaler.fit_transform(X_train)
X_val_scaled = feature_scaler.transform(X_val)
X_test_scaled = feature_scaler.transform(X_test)
X_blind_scaled = feature_scaler.transform(X_blind)

target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train)

# ======================================================
# Grid Search parameters for LightGBM
# ======================================================
param_grid = { 
    'estimator__n_estimators': [100, 200, 300],
    'estimator__learning_rate': [0.05],
    'estimator__max_depth': [5],  # -1 = no limit
    'estimator__num_leaves': [31, 50, 100],
    'estimator__min_child_samples': [20, 30, 50],
    'estimator__subsample': [0.8, 1.0],
    'estimator__colsample_bytree': [0.8, 1.0]
}

# ======================================================
# Model and CV
# ======================================================
lgbm = LGBMRegressor(
    random_state=42,
    verbose=-1,  # Silenzioso
    n_jobs=-1
)
multi_lgbm = MultiOutputRegressor(lgbm)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scorer_sklearn = make_scorer(mee_scorer, greater_is_better=True)

print("Starting Grid Search for LightGBM...")
n_combinations = (len(param_grid['estimator__n_estimators']) * 
                  len(param_grid['estimator__learning_rate']) * 
                  len(param_grid['estimator__max_depth']) * 
                  len(param_grid['estimator__num_leaves']) * 
                  len(param_grid['estimator__min_child_samples']) *
                  len(param_grid['estimator__subsample']) *
                  len(param_grid['estimator__colsample_bytree']))
print(f"Testing {n_combinations} combinations")

start_time = time.time()
gs = GridSearchCV(multi_lgbm, param_grid, scoring=scorer_sklearn, cv=kf, n_jobs=-1, verbose=1)
gs.fit(X_train_scaled, y_train_scaled)
print(f"\nGrid Search completed in {time.time()-start_time:.2f}s\n")

# Best parameters
best_params = gs.best_params_
print("Best parameters:", best_params)
print(f"Best CV score: {-gs.best_score_:.4f}")  # Negativo perché usiamo -MEE

# ======================================================
# Train final model on training set
# ======================================================
best_model = gs.best_estimator_
best_model.fit(X_train_scaled, y_train_scaled)

# Predizioni su train e validation
y_tr_pred_scaled = best_model.predict(X_train_scaled)
y_val_pred_scaled = best_model.predict(X_val_scaled)

# Inverse transform per ottenere scala originale
y_tr_pred = target_scaler.inverse_transform(y_tr_pred_scaled)
y_val_pred = target_scaler.inverse_transform(y_val_pred_scaled)

# Loss su scala originale
train_loss = euclidean_distance_score(y_train, y_tr_pred)
val_loss = euclidean_distance_score(y_val, y_val_pred)

print("\n---- Training Results (original scale) ----")
print(f"Train Loss (MEE): {train_loss:.4f}")
print(f"Val Loss (MEE):   {val_loss:.4f}")

# ======================================================
# Final evaluation on test set (original scale)
# ======================================================
y_ts_pred = target_scaler.inverse_transform(best_model.predict(X_test_scaled))
test_loss = euclidean_distance_score(y_test, y_ts_pred)
test_r2 = r2_score(y_test, y_ts_pred)

print("\n---- Final Results (original scale data) ----")
print(f"Test Loss (MEE): {test_loss:.4f}")
print(f"Test R²:         {test_r2:.4f}")
# ======================================================
# Blind test predictions
# ======================================================
y_blind_pred = target_scaler.inverse_transform(best_model.predict(X_blind_scaled))
write_blind_results("LightGBM", y_blind_pred)

# Best parameters: {'estimator__colsample_bytree': 0.8, 'estimator__learning_rate': 0.05, 'estimator__max_depth': 5, 'estimator__min_child_samples': 20, 'estimator__n_estimators': 200, 'estimator__num_leaves': 31, 'estimator__subsample': 0.8}
# Best CV score: 1.3099

# ---- Training Results (original scale) ----
# Train Loss (MEE): 12.2740
# Val Loss (MEE):   21.6113

# ---- Final Results (original scale data) ----
# Test Loss (MEE): 23.1212
# Test R²:         0.4550