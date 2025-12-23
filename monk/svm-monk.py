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
    confusion_matrix, 
    classification_report,
    roc_curve,
    roc_auc_score,
)
from monks_data_loader import load_monk_data

def calculate_majority_baseline(y_train, y_test) -> Tuple[float, object]:
    """Return baseline accuracy and majority class (works with pandas Series or numpy arrays)."""
    try:
        majority_class = y_train.mode()[0]
    except Exception:
        vals, counts = np.unique(y_train, return_counts=True)
        majority_class = vals[np.argmax(counts)]
    majority_pred = np.full(len(y_test), majority_class)
    return accuracy_score(y_test, majority_pred), majority_class

def plot_confusion_matrix_heatmap(y_true, y_pred, dataset_idx: int = 1):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - Monk-{dataset_idx}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def plot_roc_curve(y_true, y_scores, dataset_idx: int = 1):
    """
    Plot ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.title(f'ROC Curve - Monk-{dataset_idx}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.show()

def get_param_grid(dataset_idx: int) -> Dict:
    """
    Return parameter grid based on dataset index.
    """
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

def nested_cross_validation(X_train, y_train, dataset_idx: int = 1, 
                           inner_cv_folds: int = 5, 
                           outer_cv_folds: int = 5) -> Tuple[np.ndarray, List[Dict]]:
    """
    Perform nested cross-validation with two levels:
    - Outer CV: estimates generalization performance
    - Inner CV: selects best hyperparameters
    
    Args:
        X_train: Training features
        y_train: Training labels
        dataset_idx: Dataset identifier (1, 2, or 3)
        inner_cv_folds: Number of folds for inner CV (hyperparameter tuning)
        outer_cv_folds: Number of folds for outer CV (performance estimation)
    
    Returns:
        nested_scores: Array of scores from outer CV
        best_params_per_fold: List of best parameters found in each outer fold
    """
    param_grid = get_param_grid(dataset_idx)
    
    # Outer CV: per stimare la performance di generalizzazione
    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=42)
    
    nested_scores = []
    best_params_per_fold = []
    
    print(f"\n{'='*80}")
    print(f"NESTED CROSS-VALIDATION - Monk-{dataset_idx}")
    print(f"Outer CV: {outer_cv_folds} folds | Inner CV: {inner_cv_folds} folds")
    print(f"{'='*80}\n")
    
    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train), 1):
        print(f"Processing Outer Fold {fold_idx}/{outer_cv_folds}...")
        
        # Split dei dati per questo fold esterno
        X_train_fold = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
        X_val_fold = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
        y_train_fold = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
        y_val_fold = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
        
        # Inner CV: GridSearchCV per la selezione degli iperparametri
        inner_cv = StratifiedKFold(n_splits=inner_cv_folds, shuffle=True, random_state=42)
        
        pipeline = Pipeline([
            ("scaler", StandardScaler()), 
            ("svc", SVC(probability=True, random_state=42))
        ])
        
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=0,
            return_train_score=True,
        )
        
        # Fit su training fold (inner CV avviene qui)
        grid_search.fit(X_train_fold, y_train_fold)
        
        # Valutazione sul validation fold (outer CV)
        best_model = grid_search.best_estimator_
        fold_score = best_model.score(X_val_fold, y_val_fold)
        
        nested_scores.append(fold_score)
        best_params_per_fold.append(grid_search.best_params_)
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Inner CV score: {grid_search.best_score_:.4f}")
        print(f"  Outer validation score: {fold_score:.4f}\n")
    
    nested_scores = np.array(nested_scores)
    
    print(f"{'='*80}")
    print(f"NESTED CV RESULTS - Monk-{dataset_idx}")
    print(f"{'='*80}")
    print(f"Mean Accuracy: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
    print(f"Min Accuracy: {nested_scores.min():.4f}")
    print(f"Max Accuracy: {nested_scores.max():.4f}")
    print(f"Scores per fold: {nested_scores}")
    print(f"{'='*80}\n")
    
    return nested_scores, best_params_per_fold


def train_final_model(X_train, y_train, X_test, y_test, dataset_idx: int = 1):
    """
    Train final model on entire training set using GridSearchCV.
    This is for final evaluation on the test set.
    
    Returns:
        gs (GridSearchCV): The fitted GridSearchCV object.
        test_acc (float): Test accuracy score.
    """
    param_grid = get_param_grid(dataset_idx)
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()), 
        ("svc", SVC(probability=True, random_state=42))
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

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
    test_acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*80}")
    print(f"FINAL MODEL EVALUATION - Monk-{dataset_idx}")
    print(f"{'='*80}")
    print(f"Best params: {gs.best_params_}")
    print(f"CV score on training set: {gs.best_score_:.4f}")
    print(f"Test set accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"{'='*80}\n")

    return gs, test_acc


def full_analysis(X_train, y_train, X_test, y_test, dataset_idx: int = 1):
    """
    Perform complete analysis including:
    1. Nested CV for unbiased performance estimation
    2. Final model training and test set evaluation
    3. All visualizations
    """
    print(f"\n{'#'*80}")
    print(f"# COMPLETE ANALYSIS FOR MONK-{dataset_idx}")
    print(f"{'#'*80}\n")
    
    # Baseline
    baseline_acc, majority_class = calculate_majority_baseline(y_train, y_test)
    print(f"Majority Voting Baseline Accuracy: {baseline_acc:.4f} (Class: {majority_class})")
    
    # 1. NESTED CROSS-VALIDATION (unbiased performance estimate)
    nested_scores, best_params_per_fold = nested_cross_validation(
        X_train, y_train, dataset_idx=dataset_idx,
        inner_cv_folds=5, outer_cv_folds=5
    )
    
    # 2. TRAIN FINAL MODEL (for test set evaluation and visualizations)
    gs, test_acc = train_final_model(X_train, y_train, X_test, y_test, dataset_idx=dataset_idx)
    
    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    
    # 3. VISUALIZATIONS
    print("Generating visualizations...\n")
    
    # Confusion matrix
    plot_confusion_matrix_heatmap(y_test, y_pred, dataset_idx=dataset_idx)
    
    # ROC curve
    y_scores = best.predict_proba(X_test)[:, 1]
    plot_roc_curve(y_test, y_scores, dataset_idx=dataset_idx)
    
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
    
    print("="*80)
    print("ANALISI SU TUTTI I DATASET MONKS CON NESTED CV")
    print("="*80)
    
    for n_monk in [1, 2, 3]:
        x_train, y_train, x_test, y_test = load_monk_data(n_monk)
        results[n_monk] = full_analysis(x_train, y_train, x_test, y_test, dataset_idx=n_monk)
    
    # Stampa riepilogo finale
    print("\n" + "="*80)
    print("RIEPILOGO RISULTATI FINALI")
    print("="*80)
    
    for n_monk in [1, 2, 3]:
        res = results[n_monk]
        print(f"\n{'--- MONK-' + str(n_monk) + ' ---':^80}")
        print(f"Baseline Accuracy: {res['baseline_acc']:.4f} (Class: {res['majority_class']})")
        print(f"\nNested CV (unbiased estimate):")
        print(f"  Mean Accuracy: {res['nested_cv_mean']:.4f} ± {res['nested_cv_std']:.4f}")
        print(f"  Scores: {res['nested_cv_scores']}")
        print(f"\nFinal Model (trained on full training set):")
        print(f"  Best Parameters: {res['final_best_params']}")
        print(f"  CV Score: {res['final_cv_score']:.4f}")
        print(f"  Test Accuracy: {res['test_acc']:.4f}")
    
    print("\n" + "="*80)
    print("CONFRONTO PRESTAZIONI")
    print("="*80)
    print(f"{'Dataset':<12} {'Baseline':<12} {'Nested CV':<20} {'Test Acc':<12}")
    print("-"*80)
    for n_monk in [1, 2, 3]:
        res = results[n_monk]
        print(f"MONK-{n_monk:<7} {res['baseline_acc']:>8.4f}    "
              f"{res['nested_cv_mean']:>8.4f} ± {res['nested_cv_std']:.4f}    "
              f"{res['test_acc']:>8.4f}")