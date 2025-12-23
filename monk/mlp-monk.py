from typing import Tuple, Dict, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, ParameterGrid, StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random

from monks_data_loader import load_monk_data

# ============================================================================
# SEED E UTILITY
# ============================================================================

def set_seed(seed: int = 42):
    """Imposta i seed per riproducibilità."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix - Monk-{dataset_idx}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def plot_learning_curve(train_errors: List[float], val_errors: List[float], 
                       dataset_idx: int = 1, title_suffix: str = ""):
    def smooth_curve(values, window):
        if len(values) < window:
            return np.array(values)
        return np.convolve(values, np.ones(window)/window, mode='valid')
    smoothing_window = 5
    # Smoothing
    train_err_smooth = smooth_curve(train_errors, smoothing_window)
    val_err_smooth = smooth_curve(val_errors, smoothing_window)
    
    train_acc_smooth = 1 - train_err_smooth
    val_acc_smooth = 1 - val_err_smooth
    
    epochs = range(1, len(train_err_smooth) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --------- Error Rate subplot ---------
    axes[0].plot(epochs, train_err_smooth, 'b-', label='Training Error', linewidth=2)
    axes[0].plot(epochs, val_err_smooth, 'r-', label='Validation Error', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Error Rate')
    axes[0].set_title(f"Error Rate - Monk-{dataset_idx}{title_suffix}")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # --------- Accuracy subplot ---------
    axes[1].plot(epochs, train_acc_smooth, 'b--', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, val_acc_smooth, 'r--', label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f"Accuracy - Monk-{dataset_idx}{title_suffix}")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# PYTORCH MLP MODEL
# ============================================================================

class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron per classificazione binaria."""
    
    def __init__(self, input_dim: int, n_units: int):
        super(MLPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, n_units),
            nn.Sigmoid(),
            nn.Linear(n_units, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x).squeeze(1)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_model(model, optimizer, train_dataloader, val_dataloader, 
                patience=15, max_epochs=200, track_errors=False):
    """Train model with early stopping and optional error tracking."""
    criterion = nn.BCELoss()
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    
    train_errors = []
    val_errors = []

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_correct = 0
        train_total = 0
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            if track_errors:
                predicted = (y_pred > 0.5).int()
                train_correct += (predicted == y_batch.int()).sum().item()
                train_total += len(y_batch)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_dataloader:
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * len(x_batch)
                
                if track_errors:
                    predicted = (y_pred > 0.5).int()
                    val_correct += (predicted == y_batch.int()).sum().item()
                    val_total += len(y_batch)
        
        val_loss /= len(val_dataloader.dataset)
        
        if track_errors:
            train_error = 1 - (train_correct / train_total)
            val_error = 1 - (val_correct / val_total)
            train_errors.append(train_error)
            val_errors.append(val_error)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    if track_errors:
        return model, train_errors, val_errors
    return model


def evaluate_model(model, dataloader):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            y_pred = model(x_batch)
            predicted = (y_pred > 0.5).int()
            correct += (predicted == y_batch.int()).sum().item()
            total += len(y_batch)
    return correct / total if total > 0 else 0


# ============================================================================
# PARAMETER GRIDS
# ============================================================================

def get_param_grid(dataset_idx: int) -> Dict:
    """Restituisce la griglia di iperparametri per ogni dataset."""
    
    if dataset_idx == 1:
        return {
            'n_units': [3, 4],
            'learning_rate': [0.5, 0.8],
            'batch_size': [1, 2],
            'momentum': [0.8, 0.9],
            'weight_decay': [0.0]
        }
    elif dataset_idx == 2:
                return {
            'n_units': [3, 4],
                'learning_rate': [ 0.2,0.3],
                'batch_size': [1,2],
                'momentum': [0.5],
                'weight_decay': [0.0, 0.001]
        }
    else:  # dataset_idx == 3
        return {
                    'n_units': [4, 5],
        'learning_rate': [0.01, 0.05],
        'batch_size': [4, 8],
        'momentum': [0.9, 0.92],
        'weight_decay': [0.0005, 0.001, 0.005]

        }


# ============================================================================
# NESTED CROSS-VALIDATION
# ============================================================================

def nested_cross_validation(X_train, y_train, dataset_idx: int,
                            inner_cv_folds: int = 3,
                            outer_cv_folds: int = 5) -> Tuple[np.ndarray, List[Dict]]:
    """
    Nested cross-validation per MLP:
    - Outer CV: stima performance di generalizzazione
    - Inner CV: selezione iperparametri
    """
    
    print(f"\n{'='*80}")
    print(f"NESTED CROSS-VALIDATION - Monk-{dataset_idx}")
    print(f"Outer CV: {outer_cv_folds} folds | Inner CV: {inner_cv_folds} folds")
    print(f"{'='*80}\n")
    
    # Converti a numpy se necessario
    X = X_train.values if hasattr(X_train, 'values') else X_train
    y = y_train.values if hasattr(y_train, 'values') else y_train
    
    param_grid = get_param_grid(dataset_idx)
    outer_cv = StratifiedKFold(n_splits=outer_cv_folds, shuffle=True, random_state=42)
    
    nested_scores = []
    best_params_per_fold = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), 1):
        print(f"Processing Outer Fold {fold_idx}/{outer_cv_folds}...")
        
        X_train_fold = X[train_idx]
        X_val_fold = X[val_idx]
        y_train_fold = y[train_idx]
        y_val_fold = y[val_idx]
        
        # Inner CV: Grid Search per trovare i migliori iperparametri
        inner_cv = StratifiedKFold(n_splits=inner_cv_folds, shuffle=True, random_state=42)
        grid = list(ParameterGrid(param_grid))
        
        best_config = None
        best_inner_score = 0
        
        for config in grid:
            inner_scores = []
            
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_fold, y_train_fold):
                X_inner_train = torch.FloatTensor(X_train_fold[inner_train_idx])
                y_inner_train = torch.FloatTensor(y_train_fold[inner_train_idx])
                X_inner_val = torch.FloatTensor(X_train_fold[inner_val_idx])
                y_inner_val = torch.FloatTensor(y_train_fold[inner_val_idx])
                
                train_dataset = TensorDataset(X_inner_train, y_inner_train)
                val_dataset = TensorDataset(X_inner_val, y_inner_val)
                
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
                
                model = MLPClassifier(input_dim=X.shape[1], n_units=config['n_units'])
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=config['learning_rate'],
                    momentum=config['momentum'],
                    weight_decay=config['weight_decay']
                )
                if dataset_idx == 1:
                    patience = 10
                    max_epochs = 100
                else:
                    patience = 15
                    max_epochs = 200
                model = train_model(model, optimizer, train_loader, val_loader, 
                                  patience=patience, max_epochs=max_epochs, track_errors=False)
                inner_score = evaluate_model(model, val_loader)
                inner_scores.append(inner_score)
            
            avg_inner_score = np.mean(inner_scores)
            
            if avg_inner_score > best_inner_score:
                best_inner_score = avg_inner_score
                best_config = config
        
        # Train final model per questo outer fold con i migliori parametri
        X_train_t = torch.FloatTensor(X_train_fold)
        y_train_t = torch.FloatTensor(y_train_fold)
        X_val_t = torch.FloatTensor(X_val_fold)
        y_val_t = torch.FloatTensor(y_val_fold)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        
        train_loader = DataLoader(train_dataset, batch_size=best_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=best_config['batch_size'], shuffle=False)
        
        final_model = MLPClassifier(input_dim=X.shape[1], n_units=best_config['n_units'])
        optimizer = optim.SGD(
            final_model.parameters(),
            lr=best_config['learning_rate'],
            momentum=best_config['momentum'],
            weight_decay=best_config['weight_decay']
        )
        
        if dataset_idx == 1:
            patience = 10
            max_epochs = 100
        else:
            patience = 15
            max_epochs = 200
        final_model = train_model(final_model, optimizer, train_loader, val_loader, 
                                 patience=patience, max_epochs=max_epochs, track_errors=False)
        fold_score = evaluate_model(final_model, val_loader)
        
        nested_scores.append(fold_score)
        best_params_per_fold.append(best_config)
        
        print(f"  Best params: {best_config}")
        print(f"  Inner CV score: {best_inner_score:.4f}")
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


# ============================================================================
# FINAL MODEL TRAINING
# ============================================================================

def train_final_model(X_train, y_train, X_test, y_test, dataset_idx: int):
    """Train final model su tutto il training set."""
    
    print(f"\n{'='*80}")
    print(f"FINAL MODEL TRAINING - Monk-{dataset_idx}")
    print(f"{'='*80}\n")
    
    # Converti a numpy
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
    
    # Split per validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_np, y_train_np, test_size=0.2, random_state=42, stratify=y_train_np
    )
    
    # Grid search
    param_grid = get_param_grid(dataset_idx)
    grid = list(ParameterGrid(param_grid))
    
    best_config = None
    best_val_acc = 0
    best_model = None
    best_train_errors = None
    best_val_errors = None
    
    print(f"Testing {len(grid)} configurations...")
    
    for i, config in enumerate(grid):
        X_train_t = torch.FloatTensor(X_train_split)
        y_train_t = torch.FloatTensor(y_train_split)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        model = MLPClassifier(input_dim=X_train_np.shape[1], n_units=config['n_units'])
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
        if dataset_idx == 1:
            patience = 10
            max_epochs = 100
        else:
            patience = 15
            max_epochs = 200
        model, train_errors, val_errors = train_model(model, optimizer, train_loader, val_loader, 
                                                      patience=patience, max_epochs=max_epochs, track_errors=True)
        val_acc = evaluate_model(model, val_loader)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = config
            best_model = model
            best_train_errors = train_errors
            best_val_errors = val_errors
        
        if (i + 1) % 5 == 0:
            print(f"  Tested {i+1}/{len(grid)} - Best so far: {best_val_acc:.4f}")
    
    print(f"\nBest config: {best_config}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Plot learning curve del miglior modello
    if best_train_errors and best_val_errors:
        plot_learning_curve(best_train_errors, best_val_errors, 
                          dataset_idx=dataset_idx, title_suffix=" (Best Model)")
    
    # Evaluate on test set
    X_test_t = torch.FloatTensor(X_test_np)
    y_test_t = torch.FloatTensor(y_test_np)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    test_acc = evaluate_model(best_model, test_loader)
    
    # Get predictions
    best_model.eval()
    with torch.no_grad():
        y_pred = best_model(X_test_t).cpu().numpy()
        y_pred_binary = (y_pred > 0.5).astype(int)
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_np, y_pred_binary))
    print(f"{'='*80}\n")
    
    return best_model, best_config, test_acc, y_pred_binary


# ============================================================================
# FULL ANALYSIS
# ============================================================================

def full_analysis(X_train, y_train, X_test, y_test, dataset_idx: int):
    """Analisi completa con nested CV e modello finale."""
    
    print(f"\n{'#'*80}")
    print(f"# COMPLETE ANALYSIS FOR MONK-{dataset_idx}")
    print(f"{'#'*80}\n")
    
    # Baseline
    baseline_acc, majority_class = calculate_majority_baseline(y_train, y_test)
    print(f"Majority Voting Baseline: {baseline_acc:.4f} (Class: {majority_class})")
    
    # Nested CV
    nested_scores, best_params_per_fold = nested_cross_validation(
        X_train, y_train, dataset_idx=dataset_idx,
        inner_cv_folds=3, outer_cv_folds=5
    )
    
    # Final Model
    best_model, best_config, test_acc, y_pred = train_final_model(
        X_train, y_train, X_test, y_test, dataset_idx=dataset_idx
    )
    
    # Visualizations
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
    plot_confusion_matrix_heatmap(y_test_np, y_pred, dataset_idx=dataset_idx)
    
    return {
        'baseline_acc': baseline_acc,
        'majority_class': majority_class,
        'nested_cv_scores': nested_scores,
        'nested_cv_mean': nested_scores.mean(),
        'nested_cv_std': nested_scores.std(),
        'best_params_per_fold': best_params_per_fold,
        'final_best_params': best_config,
        'test_acc': test_acc,
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    set_seed(42)
    
    print("="*80)
    print("ANALISI MLP (PyTorch) CON NESTED CV SU TUTTI I DATASET MONKS")
    print("="*80)
    
    results = {}
    
    for n_monk in [1, 2, 3]:
        x_train, y_train, x_test, y_test = load_monk_data(n_monk)
        results[n_monk] = full_analysis(x_train, y_train, x_test, y_test, dataset_idx=n_monk)
    
    # Riepilogo finale
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
        print(f"\nFinal Model:")
        print(f"  Best Parameters: {res['final_best_params']}")
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