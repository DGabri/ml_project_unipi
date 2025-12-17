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
    validation_curve,
)
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    roc_curve,
    roc_auc_score,
    f1_score
)

from monks_data_loader import load_monk_data


def calculate_majority_baseline(y_train, y_test) -> Tuple[float, object]:
    """Return baseline accuracy and majority class."""
    try:
        majority_class = y_train.mode()[0]
    except Exception:
        vals, counts = np.unique(y_train, return_counts=True)
        majority_class = vals[np.argmax(counts)]
    majority_pred = np.full(len(y_test), majority_class)
    return accuracy_score(y_test, majority_pred), majority_class


def plot_confusion_matrix_heatmap(y_true, y_pred, dataset_idx: int = 1):
    """Plot confusion matrix heatmap for Logistic Regression."""
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - Logistic Regression - Monk-{dataset_idx}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def plot_validation_curve_C_lr(best, dataset_idx: int = 1, X_train=None, y_train=None):
    """Plot validation curve for hyperparameter C in Logistic Regression."""
    param_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    train_scores, test_scores = validation_curve(
        best,
        X_train,
        y_train,
        param_name="lr__C",
        param_range=param_range,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(param_range, train_mean, label="Training score", color="blue", marker='o')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.2)
    plt.plot(param_range, test_mean, label="Validation score", color="orange", marker='o')
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="orange", alpha=0.2)
    plt.title(f"Validation Curve for C - Logistic Regression - Monk-{dataset_idx}")
    plt.xlabel("C (Regularization Parameter)")
    plt.ylabel("Accuracy")
    plt.xscale("log")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_accuracy_comparison(baseline_acc: float, lr_acc: float, dataset_idx: int = 1):
    """Plot comparison of baseline and logistic regression accuracy."""
    labels = ['Majority\nVoting', 'Logistic\nRegression']
    accuracies = [baseline_acc, lr_acc]
    colors = ['#ff9999', '#ffcc99']

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    plt.title(f"Accuracy Comparison - Monk-{dataset_idx}", fontsize=14, fontweight='bold')
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1.05)
    
    for bar, v in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{v:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Calculate improvement
    improvement = ((lr_acc - baseline_acc) / baseline_acc) * 100
    plt.text(0.5, 0.5, f'Improvement: {improvement:+.2f}%', 
             transform=plt.gca().transAxes,
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(X_test, y_test, lr_pipeline, dataset_idx: int = 1):
    """Plot ROC curve for Logistic Regression."""
    y_scores = lr_pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    auc = roc_auc_score(y_test, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {auc:.4f})', linewidth=2, color='orange')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.title(f'ROC Curve - Logistic Regression - Monk-{dataset_idx}', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def train_logistic_regression_grid_search(X_train, y_train, X_test, y_test, dataset_idx: int = 1):
    """
    Train Logistic Regression with GridSearchCV and show results.
    
    Returns:
        gs: The fitted GridSearchCV object
        test_acc: Test accuracy score
        nested_scores: Nested cross-validation scores
    """
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
        verbose=0,
        return_train_score=True,
    )
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"LOGISTIC REGRESSION RESULTS - MONK-{dataset_idx}")
    print(f"{'='*60}")
    print("Best params:", gs.best_params_)
    print(f"Best CV score: {gs.best_score_:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    plot_confusion_matrix_heatmap(y_test, y_pred, dataset_idx=dataset_idx)
    
    # Validation curve for C
    plot_validation_curve_C_lr(best, dataset_idx=dataset_idx, X_train=X_train, y_train=y_train)
    
    # ROC curve
    plot_roc_curve(X_test, y_test, best, dataset_idx=dataset_idx)

    # Nested CV
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    nested_scores = cross_val_score(gs, X_train, y_train, cv=outer_cv, scoring='accuracy', n_jobs=-1)
    print(f"\nNested CV accuracy: mean={nested_scores.mean():.4f} std={nested_scores.std():.4f}")

    return gs, test_acc, nested_scores


if __name__ == "__main__":
    results = {}
    
    print("="*80)
    print("LOGISTIC REGRESSION ANALYSIS - MONKS DATASETS")
    print("="*80)
    
    for n_monk in [1, 2, 3]:
        print(f"\n{'#'*80}")
        print(f"PROCESSING MONK-{n_monk}")
        print(f"{'#'*80}")
        
        # Load data
        x_train, y_train, x_test, y_test = load_monk_data(n_monk)

        # Majority voting baseline
        baseline_acc, majority_class = calculate_majority_baseline(y_train, y_test)
        print(f"\n{'='*60}")
        print(f"MAJORITY VOTING BASELINE - MONK-{n_monk}")
        print(f"{'='*60}")
        print(f"Accuracy: {baseline_acc:.4f} (Always predicts class: {majority_class})")

        # Train Logistic Regression with GridSearch
        gs, test_acc, nested_scores = train_logistic_regression_grid_search(
            x_train, y_train, x_test, y_test, dataset_idx=n_monk
        )
        
        # Comparison plot
        plot_accuracy_comparison(baseline_acc, test_acc, dataset_idx=n_monk)
        
        # Store results
        results[n_monk] = {
            'baseline_acc': baseline_acc,
            'majority_class': majority_class,
            'best_params': gs.best_params_,
            'best_cv_score': gs.best_score_,
            'test_acc': test_acc,
            'nested_cv_mean': nested_scores.mean(),
            'nested_cv_std': nested_scores.std()
        }
    
    # FINAL SUMMARY
    print("\n" + "="*80)
    print("RIEPILOGO RISULTATI FINALI - LOGISTIC REGRESSION")
    print("="*80)
    
    for n_monk in [1, 2, 3]:
        res = results[n_monk]
        print(f"\n{'*'*60}")
        print(f"MONK-{n_monk}")
        print(f"{'*'*60}")
        print(f"Baseline (Majority): {res['baseline_acc']:.4f} (Class: {res['majority_class']})")
        print(f"Best Parameters: {res['best_params']}")
        print(f"Best CV Score: {res['best_cv_score']:.4f}")
        print(f"Test Accuracy: {res['test_acc']:.4f}")
        print(f"Nested CV: {res['nested_cv_mean']:.4f} ± {res['nested_cv_std']:.4f}")
        
        improvement = ((res['test_acc'] - res['baseline_acc']) / res['baseline_acc']) * 100
        print(f"Improvement over baseline: {improvement:+.2f}%")
    
    # COMPARATIVE TABLE
    print("\n" + "="*80)
    print("TABELLA COMPARATIVA")
    print("="*80)
    print(f"{'Dataset':<10} {'Baseline':<12} {'LR Test':<12} {'LR CV':<15}")
    print("-"*80)
    for n_monk in [1, 2, 3]:
        res = results[n_monk]
        print(f"MONK-{n_monk:<5} {res['baseline_acc']:>8.4f}    "
              f"{res['test_acc']:>8.4f}    "
              f"{res['nested_cv_mean']:>8.4f} ± {res['nested_cv_std']:.4f}")
    
    print("\n" + "="*80)
    print("ANALISI COMPLETATA")
    print("="*80)