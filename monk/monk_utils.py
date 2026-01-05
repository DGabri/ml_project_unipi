from sklearn.metrics import confusion_matrix, mean_squared_error, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from typing import Tuple
import seaborn as sns
import numpy as np
import random
import torch
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
def calculate_majority_baseline(y_train, y_test):
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

import matplotlib.pyplot as plt

def plot_curves(train_mse, val_mse, train_acc, val_acc, dataset_num, suffix=""):
    """
    Plot training and validation curves for both MSE and accuracy
    
    Args:
        train_mse: list of training MSE values
        val_mse: list of validation MSE values
        train_acc: list of training accuracy values
        val_acc: list of validation accuracy values
        dataset_num: dataset number (1, 2, or 3)
        suffix: optional suffix for the title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot MSE
    epochs = range(1, len(train_mse) + 1)
    ax1.plot(epochs, train_mse, 'b-', label='Training', linewidth=2)
    ax1.plot(epochs, val_mse, 'r-', label='Validation', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title(f'Monk-{dataset_num}: Training and Validation MSE{suffix}', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot Accuracy
    ax2.plot(epochs, train_acc, 'b--', label='Training', linewidth=2)
    ax2.plot(epochs, val_acc, 'r--', label='Validation', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'Monk-{dataset_num}: Training and Validation Accuracy{suffix}', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'monk{dataset_num}_curves{suffix.replace(" ", "_")}.png', dpi=300)
    plt.show()
    plt.close()
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_roc_curve(y_true, y_scores, title):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.show()
    
def plot_combined_C_gamma_accurate(gs: GridSearchCV, dataset_idx=1):
    """
    Grafico combinato heatmap C-gamma e accuracy vs C coerente con la nested CV.
    gs: GridSearchCV già fit
    """
    # Estrai parametri e punteggi
    mean_scores = gs.cv_results_['mean_test_score']
    params = gs.cv_results_['params']

    # Trova tutti i valori unici di C
    C_values = sorted(list({p['svc__C'] for p in params}))

    # Trova tutti i valori unici di gamma, gestendo numeri e stringhe separatamente
    gamma_set = {p['svc__gamma'] for p in params}
    gamma_num = sorted([g for g in gamma_set if isinstance(g, (int, float))])
    gamma_str = [g for g in gamma_set if isinstance(g, str)]
    gamma_values = gamma_num + gamma_str  # numeri prima, stringhe (es. 'auto') dopo

    # Costruisci matrice score (righe=C, colonne=gamma)
    scores_matrix = np.zeros((len(C_values), len(gamma_values)))
    for i, C in enumerate(C_values):
        for j, gamma in enumerate(gamma_values):
            for k, p in enumerate(params):
                if p['svc__C'] == C and p['svc__gamma'] == gamma:
                    scores_matrix[i, j] = mean_scores[k]

    # Miglior combinazione C-gamma
    best_idx = np.unravel_index(np.argmax(scores_matrix), scores_matrix.shape)
    best_C = C_values[best_idx[0]]
    best_gamma = gamma_values[best_idx[1]]
    best_acc = scores_matrix[best_idx]

    # Accuracy vs C (massimo gamma per ciascun C)
    acc_per_C = [scores_matrix[i, :].max() for i in range(len(C_values))]

    # Plots combinati
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
    fig.suptitle(f"SVM Hyperparameter Analysis - MONK-{dataset_idx}", fontsize=16)

    # Heatmap
    sns.heatmap(scores_matrix, xticklabels=[str(g) for g in gamma_values],
                yticklabels=np.round(C_values,3), cmap="viridis", ax=ax1)
    ax1.set_xlabel("gamma")
    ax1.set_ylabel("C")
    ax1.set_title(f"Heatmap CV Accuracy\nBest C={best_C}, gamma={best_gamma}")

    # Accuracy vs C
    ax2.plot(C_values, acc_per_C, marker='o', color='blue', label='CV Accuracy')
    ax2.scatter(best_C, best_acc, color='red', s=100, label=f'Best C={best_C}, gamma={best_gamma}')
    ax2.set_xlabel("C")
    ax2.set_ylabel("CV Accuracy")
    ax2.set_xticks(C_values)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_title(f"Accuracy vs C (Best gamma per C)")
    ax2.legend()

    plt.tight_layout()
    plt.show()

    print(f"Best parameters from GridSearch: C={best_C}, gamma={best_gamma}, CV Accuracy={best_acc:.4f}")


