from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
)

from monks_data_loader import load_monk_data
from monk_utils import calculate_majority_baseline, plot_confusion_matrix
def train_logistic_regression_grid_search(X_train, y_train, X_test, y_test, dataset_idx: int = 1):
    """Train Logistic Regression with GridSearchCV and show results."""
    
    # Define param grid
    param_grid = {
        'lr__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'lr__penalty': ['l2'],
        'lr__solver': ['lbfgs', 'liblinear'],
        'lr__max_iter': [1000, 2000]
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(random_state=42))
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    y_pred_test = best.predict(X_test)
    y_pred_train = best.predict(X_train)
    
    test_acc = accuracy_score(y_test, y_pred_test)
    tr_acc = accuracy_score(y_train, y_pred_train)
    cv_mean = gs.best_score_

    # Baseline
    baseline_acc, majority_class = calculate_majority_baseline(y_train, y_test)
    print(f"Baseline accuracy: {baseline_acc:.2f} (class: {majority_class})")
    
    print("")
    print(f"Best parameters: {gs.best_params_}")
    print(f"Best CV accuracy: {(cv_mean*100):.2f} %")
    
    print("")
    print(f"TR accuracy: {(tr_acc*100):.2f} %")
    print(f"Test accuracy: {(test_acc*100):.2f} %")
    print(f"Train Validation delta: {((tr_acc - test_acc)*100):.2f} %")
    print(f"CV Mean: {(cv_mean*100):.2f} %")
    
    print("")
    print(classification_report(y_test, y_pred_test))
    
    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred_test, title=f"Confusion Matrix - Monk-{dataset_idx}")

    return gs, test_acc


if __name__ == "__main__":
    results = {}
    
    for n_monk in [1, 2, 3]:
        print(f"Monk dataset id: {n_monk}")
        
        # Load data
        x_train, y_train, x_test, y_test = load_monk_data(n_monk)

        # Train Logistic Regression with GridSearch
        gs, test_acc = train_logistic_regression_grid_search(
            x_train, y_train, x_test, y_test, dataset_idx=n_monk
        )
        
        # Store results
        results[n_monk] = {
            'best_params': gs.best_params_,
            'best_cv_score': gs.best_score_,
            'test_acc': test_acc,
        }