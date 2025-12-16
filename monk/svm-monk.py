from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    validation_curve,
    LearningCurveDisplay,
    ShuffleSplit
)
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score
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


def plot_class_distribution(y, title="Class Distribution"):
    """Simple bar plot of class counts (works with pandas Series or numpy arrays)."""
    try:
        counts = y.value_counts().sort_index()
        labels = counts.index.astype(str)
        values = counts.values
    except Exception:
        vals, counts = np.unique(y, return_counts=True)
        labels = vals.astype(str)
        values = counts

    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=values, palette="pastel")
    plt.title(title)
    plt.xlabel("Classe")
    plt.ylabel("Numero di esempi")
    plt.show()

def plot_confusion_matrix_heatmap(y_true, y_pred, dataset_idx: int = 1):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    labels = np.unique(np.concatenate([y_test, y_pred]))
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - Monk-{dataset_idx}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

'''def plot_learning_curve_single(best,dataset_idx: int = 1, X_train=None, y_train=None):
    """
    Plot learning curve for a single estimator on the provided axis.
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), sharey=True)
    ax.set_title(f"Learning Curve for Monk-{dataset_idx}")

    common_params = {
        "X": X_train,
        "y": y_train,
        "train_sizes": np.linspace(0.1, 1.0),
        "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0),
        "score_type": "both",
        "n_jobs": -1,
        "line_kw": {"marker": "o"},
        "std_display_style": "fill_between",
        "score_name": "Accuracy",
    }

    # We only have one estimator to draw, so call from_estimator once
    estimator = best
    LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ["Training Score", "Validation Score"])
    ax.set_title(f"Learning Curve for {estimator.__class__.__name__}")  
    plt.suptitle(f"Learning Curves - Monk-{dataset_idx}")
    plt.show()
'''

# plot validation curve su C
def plot_validation_curve_C(best, dataset_idx: int = 1, X_train=None, y_train=None):
    """
    Plot validation curve for hyperparameter C.
    """
    param_range = [0.01, 0.1, 1, 10, 20, 30, 50]
    train_scores, test_scores = validation_curve(
        best,
        X_train,
        y_train,
        param_name="svc__C",
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
    plt.plot(param_range, train_mean, label="Training score", color="blue")
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.2)
    plt.plot(param_range, test_mean, label="Validation score", color="orange")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="orange", alpha=0.2)
    plt.title(f"Validation Curve for C - Monk-{dataset_idx}")
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.xscale("log")
    plt.legend(loc="best")
    plt.show()

# plot validation curve su gamma
def plot_validation_curve_gamma(best, dataset_idx: int = 1, X_train=None, y_train=None):
    """
    Plot validation curve for hyperparameter gamma.
    """ 
    param_range = [0.001, 0.01, 0.1, 1, 10]
    train_scores, test_scores = validation_curve(
        best,
        X_train,
        y_train,
        param_name="svc__gamma",
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
    plt.plot(param_range, train_mean, label="Training score", color="blue")
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.2)
    plt.plot(param_range, test_mean, label="Validation score", color="orange")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="orange", alpha=0.2)
    plt.title(f"Validation Curve for Gamma - Monk-{dataset_idx}")
    plt.xlabel("Gamma")
    plt.ylabel("Accuracy")
    plt.xscale("log")
    plt.legend(loc="best")
    plt.show()

# funzione che faccia il confronto accuracy tra baseline e svm 
def plot_accuracy_comparison(baseline_acc: float, svm_acc: float, dataset_idx: int = 1):
    """
    Plot comparison of baseline accuracy and SVM accuracy.
    """
    labels = ['Baseline', 'SVM']
    accuracies = [baseline_acc, svm_acc]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=labels, y=accuracies, palette="pastel")
    plt.title(f"Accuracy Comparison - Monk-{dataset_idx}")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
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

def train_svm_grid_search(X_train, y_train, X_test, y_test, dataset_idx: int = 1):
    """
    Train SVM with GridSearchCV and show results:
      - best params
      - test accuracy + classification report
      - confusion matrix heatmap
      - validation curve on C
      - learning curve
      - nested CV score (outer CV)
    
    Returns:
        gs (GridSearchCV): The fitted GridSearchCV object.
        test_acc (float): Test accuracy score.
        nested_scores (np.ndarray): Nested cross-validation scores.
    """
    # define param grid per dataset index
    if dataset_idx == 1:
        param_grid = {
                "svc__kernel": ["linear","rbf"],
                "svc__C": [0.1, 1, 10, 20, 30, 50],
                "svc__gamma": [0.01, 0.1, 1],
        }
    elif dataset_idx == 2:
       param_grid =  {
            'svc__kernel': ['poly',"linear","rbf"],
            'svc__degree': [2,3],                          # quello che ha dato i migliori risultati
            'svc__C': [ 0.1, 1, 5, 10],
           "svc__gamma": [ "auto", 0.001, 0.01],
                     # quello che ha funzionato
            'svc__class_weight': ['balanced'],           # perché classe sbilanciata
    }
    else:
        param_grid = { 
            "svc__kernel": ["linear",],
            "svc__C": [0.05, 0.1, 0.2,1,2,3,4,5,10],  # piccoli valori vicino al best che già funziona
            "svc__gamma": [ "auto", 0.001, 0.01, 0.1, 1],
            'svc__class_weight': ['balanced'],  # perché classe sbilanciata
        }

    pipeline = Pipeline(
        [("scaler", StandardScaler()), ("svc", SVC(probability=True, random_state=42))]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    gs = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
        error_score="raise",
    )
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"\nDataset: Monk-{dataset_idx}")
    print("Best params:", gs.best_params_)
    print(f"Test accuracy: {test_acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    # confusion matrix heatmap
    plot_confusion_matrix_heatmap(y_test, y_pred, dataset_idx=dataset_idx)
    # Learning curve
    # plot_learning_curve_single(best,dataset_idx=dataset_idx, X_train=X_train, y_train=y_train)
    # Validation curve C
    plot_validation_curve_C(best,dataset_idx=dataset_idx, X_train=X_train, y_train=y_train)
    # Validation curve gamma
    if gs.best_params_['svc__kernel'] == 'rbf':
        plot_validation_curve_gamma(best,dataset_idx=dataset_idx, X_train=X_train, y_train=y_train)
    # Accuracy comparison
    plot_accuracy_comparison(calculate_majority_baseline(y_train, y_test)[0], test_acc, dataset_idx=dataset_idx)
    # ROC curve
    y_scores = best.predict_proba(X_test)[:, 1]  # Probabilità della classe positiva
    plot_roc_curve(y_test, y_scores, dataset_idx=dataset_idx)

    # Nested CV (outer)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    nested_scores = cross_val_score(gs, X_train, y_train, cv=outer_cv, scoring="accuracy", n_jobs=-1)
    print("Nested CV accuracy: mean=%.4f std=%.4f" % (nested_scores.mean(), nested_scores.std()))

    return gs, test_acc, nested_scores


if __name__ == "__main__":
    results = {}
    
    print("="*80)
    print("ANALISI SU TUTTI I DATASET MONKS")
    print("="*80)
    
    for n_monk in [1, 2, 3]:
        print(f"\n{'='*80}")
        print(f"PROCESSING MONK-{n_monk}")
        print(f"{'='*80}")
        
        x_train, y_train, x_test, y_test = load_monk_data(n_monk)

        baseline_acc, majority_class = calculate_majority_baseline(y_train, y_test)
        print(f"Majority Voting Baseline Accuracy: {baseline_acc:.4f} (Class: {majority_class})")

        gs, test_acc, nested_scores = train_svm_grid_search(x_train, y_train, x_test, y_test, dataset_idx=n_monk)
        
        results[n_monk] = {
            'baseline_acc': baseline_acc,
            'majority_class': majority_class,
            'best_params': gs.best_params_,
            'test_acc': test_acc,
            'nested_cv_mean': nested_scores.mean(),
            'nested_cv_std': nested_scores.std()
        }
    
    # Stampa riepilogo finale
    print("\n" + "="*80)
    print("RIEPILOGO RISULTATI FINALI")
    print("="*80)
    
    for n_monk in [1, 2, 3]:
        res = results[n_monk]
        print(f"\n--- MONK-{n_monk} ---")
        print(f"Baseline Accuracy (Majority Class): {res['baseline_acc']:.4f} (Class: {res['majority_class']})")
        print(f"Best Parameters: {res['best_params']}")
        print(f"Test Accuracy: {res['test_acc']:.4f}")
        print(f"Nested CV Accuracy: {res['nested_cv_mean']:.4f} ± {res['nested_cv_std']:.4f}")
    
    print("\n" + "="*80)
    print("CONFRONTO PRESTAZIONI")
    print("="*80)
    print(f"{'Dataset':<10} {'Baseline':<12} {'Test Acc':<12} {'Nested CV':<15}")
    print("-"*80)
    for n_monk in [1, 2, 3]:
        res = results[n_monk]
        print(f"MONK-{n_monk:<5} {res['baseline_acc']:>8.4f}    {res['test_acc']:>8.4f}    {res['nested_cv_mean']:>8.4f} ± {res['nested_cv_std']:.4f}")