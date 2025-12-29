from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from typing import Tuple
import seaborn as sns
import numpy as np
import random
import torch

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

def plot_curves(train_errs, val_errs, dataset_num=1, suffix=""):
    def smooth(vals, w=5):
        if len(vals) < w:
            return np.array(vals)
        return np.convolve(vals, np.ones(w)/w, mode='valid')
    
    tr_smooth = smooth(train_errs)
    val_smooth = smooth(val_errs)
    
    tr_acc = 1 - tr_smooth
    val_acc = 1 - val_smooth
    
    epochs = range(1, len(tr_smooth) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, tr_smooth, 'b-', label='Train Error', linewidth=2)
    ax1.plot(epochs, val_smooth, 'r-', label='Val Error', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Error')
    ax1.set_title(f"Error - Monk-{dataset_num}{suffix}")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(epochs, tr_acc, 'b--', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_acc, 'r--', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f"Accuracy - Monk-{dataset_num}{suffix}")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

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