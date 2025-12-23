import time
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, make_scorer
from warnings import filterwarnings
from ml_cup_data_loader import (
    load_training_set,
    load_test_set,
    euclidean_distance_score,
    mee_scorer,
    write_blind_results,
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

print(f"Total dataset size: {X.shape[0]} samples")
print(f"Features: {X.shape[1]}, Targets: {y.shape[1]}\n")

# ======================================================
# Hyperparameter Grid
# ======================================================
param_grid = {
    'estimator__kernel': ['rbf'],
    'estimator__C': [5,6,8, 10, 12, 15],
    'estimator__gamma': [0.5,0.8,1.0,1.2,1.5],
    'estimator__epsilon': [0.25, 0.3, 0.35,0.4,0.5]
}

n_combinations = (len(param_grid['estimator__C']) * 
                  len(param_grid['estimator__gamma']) * 
                  len(param_grid['estimator__epsilon']))

print(f"Hyperparameter combinations to test: {n_combinations}")

# ======================================================
# Nested Cross-Validation Setup
# ======================================================
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=123)

print(f"Outer CV: {outer_cv.n_splits}-Fold")
print(f"Inner CV: {inner_cv.n_splits}-Fold")
print(f"Total models to train: ~{outer_cv.n_splits * inner_cv.n_splits * n_combinations}\n")

# ======================================================
# Storage for results
# ======================================================
outer_fold_results = []
best_params_per_fold = []

# ======================================================
# NESTED CROSS-VALIDATION
# ======================================================
print("="*70)
print("STARTING NESTED CROSS-VALIDATION")
print("="*70)

total_start_time = time.time()

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
    print(f"\n{'='*70}")
    print(f"OUTER FOLD {fold_idx}/{outer_cv.n_splits}")
    print(f"{'='*70}")
    
    fold_start_time = time.time()
    
    # Split data for this outer fold
    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]
    
    print(f"Train size: {X_train_outer.shape[0]}, Test size: {X_test_outer.shape[0]}")
    
    # Scale features and targets
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train_outer)
    X_test_scaled = feature_scaler.transform(X_test_outer)
    
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train_outer)
    
    # Custom scorer for original scale
    def mee_scorer_unscaled(y_true_scaled, y_pred_scaled):
        y_true = target_scaler.inverse_transform(y_true_scaled)
        y_pred = target_scaler.inverse_transform(y_pred_scaled)
        return mee_scorer(y_true, y_pred)
    
    scorer_unscaled = make_scorer(mee_scorer_unscaled, greater_is_better=True)
    
    # ======================================================
    # INNER LOOP: Model Selection with Grid Search
    # ======================================================
    print(f"\nRunning Inner Grid Search (5-Fold CV)...")
    
    svr = SVR()
    multi_svr = MultiOutputRegressor(svr)
    
    gs = GridSearchCV(
        multi_svr,
        param_grid,
        scoring=scorer_unscaled,
        cv=inner_cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    
    gs.fit(X_train_scaled, y_train_scaled)
    
    best_params = gs.best_params_
    best_params_per_fold.append(best_params)
    
    print(f"Best params for fold {fold_idx}: {best_params}")
    
    # Inner CV scores
    inner_val_score = -gs.best_score_  # MEE (lower is better)
    print(f"Inner CV Validation Score (MEE): {inner_val_score:.6f}")
    
    # ======================================================
    # OUTER LOOP: Model Assessment
    # ======================================================
    best_model = gs.best_estimator_
    
    # Training error on outer train set
    y_train_pred_scaled = best_model.predict(X_train_scaled)
    y_train_pred = target_scaler.inverse_transform(y_train_pred_scaled)
    train_error = euclidean_distance_score(y_train_outer, y_train_pred)
    
    # Test error on outer test set
    y_test_pred_scaled = best_model.predict(X_test_scaled)
    y_test_pred = target_scaler.inverse_transform(y_test_pred_scaled)
    test_error = euclidean_distance_score(y_test_outer, y_test_pred)
    test_r2 = r2_score(y_test_outer, y_test_pred)
    
    print(f"\nOuter Fold {fold_idx} Results:")
    print(f"  Training Error (MEE):   {train_error:.6f}")
    print(f"  Test Error (MEE):       {test_error:.6f}")
    print(f"  Test R²:                {test_r2:.6f}")
    
    outer_fold_results.append({
        'fold': fold_idx,
        'train_error': train_error,
        'test_error': test_error,
        'test_r2': test_r2,
        'best_params': best_params
    })
    
    fold_time = time.time() - fold_start_time
    print(f"\nFold {fold_idx} completed in {fold_time:.2f} seconds")

total_time = time.time() - total_start_time

# ======================================================
# AGGREGATE RESULTS FROM NESTED CV
# ======================================================
print("\n" + "="*70)
print("NESTED CROSS-VALIDATION RESULTS - FINAL SUMMARY")
print("="*70)

train_errors = [r['train_error'] for r in outer_fold_results]
test_errors = [r['test_error'] for r in outer_fold_results]
test_r2s = [r['test_r2'] for r in outer_fold_results]

print("\nPer-Fold Results:")
print("-" * 70)
for result in outer_fold_results:
    print(f"Fold {result['fold']}: "
          f"Train={result['train_error']:.6f}, "
          f"Test={result['test_error']:.6f}, "
          f"R²={result['test_r2']:.6f}")

print("\n" + "="*70)
print("AGGREGATED PERFORMANCE METRICS")
print("="*70)
print(f"Mean Training Error:   {np.mean(train_errors):.6f} ± {np.std(train_errors):.6f}")
print(f"Mean Test Error (MEE): {np.mean(test_errors):.6f} ± {np.std(test_errors):.6f}")
print(f"Mean Test R²:          {np.mean(test_r2s):.6f} ± {np.std(test_r2s):.6f}")

print("\n" + "="*70)
print("HYPERPARAMETERS SELECTED IN EACH OUTER FOLD")
print("="*70)
for i, params in enumerate(best_params_per_fold, 1):
    print(f"Fold {i}: {params}")

# Most common hyperparameters
from collections import Counter
param_keys = best_params_per_fold[0].keys()
most_common_params = {}
for key in param_keys:
    values = [p[key] for p in best_params_per_fold]
    most_common = Counter(values).most_common(1)[0][0]
    most_common_params[key] = most_common

print("\nMost Common Hyperparameters Across Folds:")
print(most_common_params)

print(f"\nTotal computation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

# ======================================================
# FINAL MODEL: Retrain on ALL data with most common params
# ======================================================
print("\n" + "="*70)
print("RETRAINING FINAL MODEL ON ENTIRE DATASET")
print("="*70)
print(f"Using hyperparameters: {most_common_params}\n")

# Scale full dataset
final_feature_scaler = StandardScaler()
X_scaled = final_feature_scaler.fit_transform(X)
X_blind_scaled = final_feature_scaler.transform(X_blind)

final_target_scaler = StandardScaler()
y_scaled = final_target_scaler.fit_transform(y)

# Create and train final model
final_svr = SVR(
    kernel=most_common_params['estimator__kernel'],
    C=most_common_params['estimator__C'],
    gamma=most_common_params['estimator__gamma'],
    epsilon=most_common_params['estimator__epsilon']
)
final_model = MultiOutputRegressor(final_svr)
final_model.fit(X_scaled, y_scaled)

# Final training error
y_pred_scaled = final_model.predict(X_scaled)
y_pred = final_target_scaler.inverse_transform(y_pred_scaled)
final_train_error = euclidean_distance_score(y, y_pred)
final_train_r2 = r2_score(y, y_pred)

print(f"Final Model Training Error (MEE): {final_train_error:.6f}")
print(f"Final Model Training R²:          {final_train_r2:.6f}")

# ======================================================
# PREDICT ON BLIND TEST SET
# ======================================================
y_blind_pred_scaled = final_model.predict(X_blind_scaled)
y_blind_pred = final_target_scaler.inverse_transform(y_blind_pred_scaled)
write_blind_results("SklearnSVM_CUP_NestedCV", y_blind_pred)

print("\n" + "="*70)
print("REPORT SUMMARY FOR ML CUP")
print("="*70)
print(f"Model Assessment Method: 5-Fold Nested Cross-Validation")
print(f"Model Selection Method:  5-Fold Inner CV with Grid Search")
print(f"\nEstimated Generalization Error (from Nested CV):")
print(f"  Mean Test Error (MEE): {np.mean(test_errors):.6f} ± {np.std(test_errors):.6f}")
print(f"  Mean Test R²:          {np.mean(test_r2s):.6f} ± {np.std(test_r2s):.6f}")
print(f"\nFinal Model (trained on all data):")
print(f"  Hyperparameters: {most_common_params}")
print(f"  Training Error:  {final_train_error:.6f}")
print(f"  Training R²:     {final_train_r2:.6f}")
print("\nBlind test predictions saved to file.")
print("="*70)