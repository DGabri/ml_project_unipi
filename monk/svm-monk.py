from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score, 
    classification_report,
    mean_squared_error,
)
from monks_data_loader import load_monk_data
from monk_utils import *

# parameter grid for each dataset
def get_param_grid(dataset_idx: int) -> Dict:
    if dataset_idx == 1:
        return {
            "svc__kernel": ["linear", "rbf"],
            "svc__C": [0.1, 1, 10, 20, 30, 50, 100],
            "svc__gamma": [0.01, 0.1, 1],
        }
    elif dataset_idx == 2:
        return {
            'svc__kernel': ['poly', "linear", "rbf"],
            'svc__degree': [2, 3],
            'svc__C': [1,5, 10, 15, 20, 25, 50, 70, 80, 90, 100],
            "svc__gamma": ["auto", 0.001, 0.01],
            'svc__class_weight': ['balanced'],
        }
    else:
        return {
            "svc__kernel": ["linear"],
            "svc__C": [0.05, 0.1, 0.2, 1, 2, 3],
            "svc__gamma": ["auto", 0.001, 0.01, 0.1, 1],
            'svc__class_weight': ['balanced'],
        }

# nested cross-validation function
def nested_cross_validation(X_train, y_train, dataset_idx: int = 1, inner_cv_folds: int = 5, outer_cv_folds: int = 5) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    param_grid = get_param_grid(dataset_idx)
    
    # MODEL ASSESSMENT
    # Outer CV
    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=42)
    
    nested_scores = []
    nested_mse = []
    best_params_per_fold = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train), 1):
        
        # split data for this fold
        X_train_fold = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
        X_val_fold = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
        y_train_fold = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
        y_val_fold = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
        
        # inner CV for hyperparameter tuning
        inner_cv = StratifiedKFold(n_splits=inner_cv_folds, shuffle=True, random_state=42)
        
        # define pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()), 
            ("svc", SVC(probability=True, random_state=42))
        ])
        
        # grid search
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
            return_train_score=True,
        )
        
        grid_search.fit(X_train_fold, y_train_fold)

        best_model = grid_search.best_estimator_
        y_pred_val = best_model.predict(X_val_fold)
        y_proba_val = best_model.predict_proba(X_val_fold)[:, 1]
        
        fold_score = accuracy_score(y_val_fold, y_pred_val)
        fold_mse = mean_squared_error(y_val_fold, y_proba_val)
        
        nested_scores.append(fold_score)
        nested_mse.append(fold_mse)
        best_params_per_fold.append(grid_search.best_params_)
        
        print(f"Fold {fold_idx}: Best params: {grid_search.best_params_} | Inner CV score: {grid_search.best_score_:.4f} | Outer acc: {fold_score:.4f} | Outer MSE: {fold_mse:.4f}")
    
    # summarize results
    nested_scores = np.array(nested_scores)
    nested_mse = np.array(nested_mse)
    
    print(f"\nNESTED CV RESULTS - Monk-{dataset_idx}")
    print(f"Accuracy: {nested_scores.mean():.4f} ± {nested_scores.std():.4f} | Range: [{nested_scores.min():.4f}, {nested_scores.max():.4f}]")
    print(f"MSE: {nested_mse.mean():.4f} ± {nested_mse.std():.4f} | Range: [{nested_mse.min():.4f}, {nested_mse.max():.4f}]")
    
    return nested_scores, nested_mse, best_params_per_fold

# train final model on full training set with best hyperparameters found
def train_final_model(X_train, y_train, X_test, y_test, dataset_idx: int = 1):
    param_grid = get_param_grid(dataset_idx)
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()), 
        ("svc", SVC(probability=True, random_state=42))
    ])
    
    # cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # grid search
    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )
    
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    
    # predictions
    y_pred_test = best.predict(X_test)
    y_pred_train = best.predict(X_train)
    
    # probabilities for MSE
    y_proba_test = best.predict_proba(X_test)[:, 1]
    y_proba_train = best.predict_proba(X_train)[:, 1]
    
    # accuracy
    test_acc = accuracy_score(y_test, y_pred_test)
    tr_acc = accuracy_score(y_train, y_pred_train)
    val_acc = gs.best_score_
    
    # MSE
    test_mse = mean_squared_error(y_test, y_proba_test)
    tr_mse = mean_squared_error(y_train, y_proba_train)
    
    # Calculate validation MSE from CV results
    # We need to compute the average MSE across all CV folds
    cv_results = gs.cv_results_
    best_index = gs.best_index_
    
    # Get predictions for validation sets during CV to compute MSE
    # Since GridSearchCV doesn't store probabilities, we need to refit on each fold
    val_mse_list = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr_fold = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
        X_val_fold = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
        y_tr_fold = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
        y_val_fold = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
        
        # Clone and fit the best model
        fold_model = gs.best_estimator_
        fold_model.fit(X_tr_fold, y_tr_fold)
        y_proba_val_fold = fold_model.predict_proba(X_val_fold)[:, 1]
        fold_mse = mean_squared_error(y_val_fold, y_proba_val_fold)
        val_mse_list.append(fold_mse)
    
    val_mse = np.mean(val_mse_list)

    return gs, test_acc, tr_acc, val_acc, test_mse, tr_mse, val_mse, y_pred_test


def full_analysis(X_train, y_train, X_test, y_test, dataset_idx: int = 1):
    
    # Baseline
    baseline_acc, majority_class = calculate_majority_baseline(y_train, y_test)
    print(f"Baseline accuracy: {baseline_acc:.2f} (class: {majority_class})")
    
    # nested cross-validation
    nested_scores, nested_mse, best_params_per_fold = nested_cross_validation(
        X_train, y_train, dataset_idx=dataset_idx,
        inner_cv_folds=5, outer_cv_folds=5
    )
    
    # train final model
    gs, test_acc, tr_acc, val_acc, test_mse, tr_mse, val_mse, y_pred = train_final_model(
        X_train, y_train, X_test, y_test, dataset_idx=dataset_idx)
    cv_mean = gs.best_score_
    best = gs.best_estimator_
    
    print("")
    print(f"Best parameters: {gs.best_params_}")
    print(f"Best CV accuracy: {(cv_mean*100):.2f} %")
    
    print("")
    print(f"TR accuracy: {(tr_acc*100):.2f} %")
    print(f"Val accuracy: {(val_acc*100):.2f} %")   
    print(f"TS accuracy: {(test_acc*100):.2f} %")
    print(f"Train-Test delta: {((tr_acc - test_acc)*100):.2f} %")
    
    print("")
    print(f"TR MSE: {tr_mse:.4f}")
    print(f"Val MSE: {val_mse:.4f}")
    print(f"TS MSE: {test_mse:.4f}")
    
    print("")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix - Monk-{dataset_idx}")
 
    plot_combined_C_gamma_accurate(gs, dataset_idx=dataset_idx)
    
    return {
        'baseline_acc': baseline_acc,
        'majority_class': majority_class,
        'nested_cv_scores': nested_scores,
        'nested_cv_mean': nested_scores.mean(),
        'nested_cv_std': nested_scores.std(),
        'nested_cv_mse': nested_mse,
        'nested_cv_mse_mean': nested_mse.mean(),
        'nested_cv_mse_std': nested_mse.std(),
        'best_params_per_fold': best_params_per_fold,
        'final_best_params': gs.best_params_,
        'final_cv_score': gs.best_score_,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'tr_acc': tr_acc,
        'val_mse': val_mse,
        'test_mse': test_mse,
        'tr_mse': tr_mse,
    }


if __name__ == "__main__":
    results = {}
    
    for n_monk in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"MONK-{n_monk} ANALYSIS")
        print(f"{'='*60}\n")
        
        x_train, y_train, x_test, y_test = load_monk_data(n_monk)
        results[n_monk] = full_analysis(x_train, y_train, x_test, y_test, dataset_idx=n_monk)