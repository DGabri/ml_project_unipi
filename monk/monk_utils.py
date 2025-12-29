from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from typing import Tuple
import seaborn as sns
import numpy as np

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