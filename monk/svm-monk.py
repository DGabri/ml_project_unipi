from typing import Tuple

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
)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from monks_data_loader import load_monk_data


def calculate_majority_baseline(y_train, y_test) -> Tuple[float, object]:
    """Return baseline accuracy and majority class (works with pandas Series or numpy arrays)."""
    try:
        # pandas Series
        majority_class = y_train.mode()[0]
    except Exception:
        # numpy array / fallback
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


def train_svm_grid_search(X_train, y_train, X_test, y_test, dataset_idx: int = 1):
    """
    Train SVM with GridSearchCV and show results:
      - best params
      - test accuracy + classification report
      - confusion matrix heatmap
      - validation curve on C
      - learning curve
      - nested CV score (outer CV)
    """
    # define param grid per dataset index
    if dataset_idx == 1:
        param_grid = {
                "svc__kernel": ["linear","rbf"],
                "svc__C": [10,20,30,40,50,60,70,80,90,100], # ampliato per curva di validazione causa plot (sopra 10 circa si stabilizza)
                "svc__gamma": ["scale", 0.01, 0.1, 1],
        }
    elif dataset_idx == 2:
       param_grid =  {
        'svc__kernel': ['poly'],
        'svc__degree': [2,3],                          # quello che ha dato i migliori risultati
        'svc__C': [1, 5, 10, 15, 20, 30,50,60, 100],  # copre entrambi i massimi osservati
        'svc__gamma': ['scale'],                     # quello che ha funzionato
        'svc__class_weight': ['balanced']            # perché classe sbilanciata
    }
    else:
        param_grid = { "svc__kernel": ["linear"],
        "svc__C": [0.05, 0.1, 0.2]  # piccoli valori vicino al best che già funziona
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
        verbose=1,
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
    # cm = confusion_matrix(y_test, y_pred)
    # labels = np.unique(np.concatenate([y_test, y_pred]))
    # plt.figure(figsize=(6, 4))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    # plt.title(f"Confusion Matrix - Monk-{dataset_idx}")
    # plt.xlabel("Predicted")
    # plt.ylabel("True")
    # plt.show()

    # Validation curve on C (fix kernel to best choice)
    # selected_kernel = gs.best_params_.get("svc__kernel", None)
    # if selected_kernel is None:
    #     # if kernel not present in best params, default to svc.kernel in estimator
    #     selected_kernel = best.named_steps["svc"].kernel

    # val_estimator = Pipeline(
    #     [("scaler", StandardScaler()), ("svc", SVC(probability=True, random_state=42, kernel=selected_kernel))]
    # )
    # param_name = "svc__C"
    # param_range = [1, 5, 10, 15, 20, 30, 50, 70, 100]
    # train_scores, val_scores = validation_curve(
    #     val_estimator, X_train, y_train, param_name=param_name, param_range=param_range, cv=cv, scoring="accuracy", n_jobs=-1
    # )
    # train_mean = np.mean(train_scores, axis=1)
    # val_mean = np.mean(val_scores, axis=1)
    # plt.figure(figsize=(8, 6))
    # plt.semilogx(param_range, train_mean, label="train")
    # plt.semilogx(param_range, val_mean, label="val")
    # plt.xlabel("C (log scale)")
    # plt.ylabel("Accuracy")
    # plt.title("Validation Curve: C")
    # plt.legend()
    # plt.grid(alpha=0.3)
    # plt.show()

    # Learning curve
    # train_sizes, train_scores_lc, val_scores_lc = learning_curve(
    #     best, X_train, y_train, cv=cv, train_sizes=np.linspace(0.1, 1.0, 5), scoring="accuracy", n_jobs=-1
    # )
    # plt.figure()
    # plt.plot(train_sizes, np.mean(train_scores_lc, axis=1), label="train")
    # plt.plot(train_sizes, np.mean(val_scores_lc, axis=1), label="val")
    # plt.xlabel("Training set size")
    # plt.ylabel("Accuracy")
    # plt.title("Learning Curve")
    # plt.legend()
    # plt.grid(alpha=0.3)
    # plt.show()

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

        # optional: plot distributions
        # plot_class_distribution(y_train, title=f"Distribuzione Classi - Training Set Monk-{n_monk}")
        # plot_class_distribution(y_test, title=f"Distribuzione Classi - Test Set Monk-{n_monk}")

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