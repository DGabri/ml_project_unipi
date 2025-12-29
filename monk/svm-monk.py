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
)
from monks_data_loader import load_monk_data
from monk_utils import plot_roc_curve, plot_confusion_matrix,calculate_majority_baseline

# parameter grid for each dataset
def get_param_grid(dataset_idx: int) -> Dict:
    if dataset_idx == 1:
        return {
            "svc__kernel": ["linear", "rbf"],
            "svc__C": [0.1, 1, 10, 20, 30, 50],
            "svc__gamma": [0.01, 0.1, 1],
        }
    elif dataset_idx == 2:
        return {
            'svc__kernel': ['poly', "linear", "rbf"],
            'svc__degree': [2, 3],
            'svc__C': [ 5, 10, 15, 20, 25, 50, 70, 80, 90, 100],
            "svc__gamma": ["auto", 0.001, 0.01],
            'svc__class_weight': ['balanced'],
        }
    else:
        return {
            "svc__kernel": ["linear"],
            "svc__C": [0.05, 0.1, 0.2, 1, 2, 3, 4, 5, 10],
            "svc__gamma": ["auto", 0.001, 0.01, 0.1, 1],
            'svc__class_weight': ['balanced'],
        }

# nested cross-validation function
def nested_cross_validation(X_train, y_train, dataset_idx: int = 1, inner_cv_folds: int = 5, outer_cv_folds: int = 5) -> Tuple[np.ndarray, List[Dict]]:
    param_grid = get_param_grid(dataset_idx)
    
    # MODEL ASSESSMENT
    # Outer CV
    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=42)
    
    nested_scores = []
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
        fold_score = best_model.score(X_val_fold, y_val_fold)
        
        nested_scores.append(fold_score)
        best_params_per_fold.append(grid_search.best_params_)
        
        print(f"Fold {fold_idx}: Best params: {grid_search.best_params_} | Inner CV score: {grid_search.best_score_:.4f} | Outer validation score: {fold_score:.4f}")
    
    # summarize results
    nested_scores = np.array(nested_scores)
    
    print(f"NESTED CV RESULTS - Monk-{dataset_idx}")
    print(f"Mean Accuracy: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
    return nested_scores, best_params_per_fold

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
    y_pred = best.predict(X_test)
    y_pred_train = best.predict(X_train)
    test_acc = accuracy_score(y_test, y_pred)
    tr_acc = accuracy_score(y_train, y_pred_train)
    y_pred_test = y_pred

    return gs, test_acc, tr_acc,y_pred_test


def full_analysis(X_train, y_train, X_test, y_test, dataset_idx: int = 1):
    
    # Baseline
    baseline_acc, majority_class = calculate_majority_baseline(y_train, y_test)
    print(f"Baseline accuracy: {baseline_acc:.2f} (class: {majority_class})")
    
    # nested cross-validation
    nested_scores, best_params_per_fold = nested_cross_validation(
        X_train, y_train, dataset_idx=dataset_idx,
        inner_cv_folds=5, outer_cv_folds=5
    )
    
    # train final model
    gs, test_acc, tr_acc, y_pred = train_final_model(X_train, y_train, X_test, y_test, dataset_idx=dataset_idx)
    cv_mean = gs.best_score_
    best = gs.best_estimator_
    
    print("")
    print(f"Best parameters: {gs.best_params_}")
    print(f"Best CV accuracy: {(cv_mean*100):.2f} %")
    
    print("")
    print(f"TR accuracy: {(tr_acc*100):.2f} %")
    print(f"Test accuracy: {(test_acc*100):.2f} %")
    print(f"Train Validation delta: {((tr_acc - test_acc)*100):.2f} %")
    print(f"CV Mean: {(cv_mean*100):.2f} %")
    
    print("")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, title=f"Confusion Matrix - Monk-{dataset_idx}")
    
    # ROC curve
    y_scores = best.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_scores,title=f"ROC Curve - Monk-{dataset_idx}")
    
    return {
        'baseline_acc': baseline_acc,
        'majority_class': majority_class,
        'nested_cv_scores': nested_scores,
        'nested_cv_mean': nested_scores.mean(),
        'nested_cv_std': nested_scores.std(),
        'best_params_per_fold': best_params_per_fold,
        'final_best_params': gs.best_params_,
        'final_cv_score': gs.best_score_,
        'test_acc': test_acc,
    }


if __name__ == "__main__":
    results = {}
    
    for n_monk in [1, 2, 3]:
        print(f"Monk dataset id: {n_monk}")
        
        x_train, y_train, x_test, y_test = load_monk_data(n_monk)
        results[n_monk] = full_analysis(x_train, y_train, x_test, y_test, dataset_idx=n_monk)