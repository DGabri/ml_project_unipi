from typing import Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import itertools

from monks_data_loader import load_monk_data


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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
    plt.show()


def plot_learning_curve_single(train_losses: List[float], val_losses: List[float], train_accs: List[float], val_accs: List[float],dataset_idx: int = 1):
    """Plot learning curve for MLP."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Training Loss', marker='o', markersize=3)
    ax1.plot(val_losses, label='Validation Loss', marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss Curves - Monk-{dataset_idx}')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Training Accuracy', marker='o', markersize=3)
    ax2.plot(val_accs, label='Validation Accuracy', marker='s', markersize=3)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Accuracy Curves - Monk-{dataset_idx}')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# plot validation curve for MLP
def plot_validation_curve(param_name: str, param_range: List, train_scores: List[float], val_scores: List[float], dataset_idx: int = 1):
    """Plot validation curve for MLP."""
    plt.figure(figsize=(8, 6))
    plt.plot(param_range, train_scores, label='Training Score', marker='o')
    plt.plot(param_range, val_scores, label='Validation Score', marker='s')
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title(f'Validation Curve - Monk-{dataset_idx}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xscale('log' if all(isinstance(x, float) and x > 0 for x in param_range) else 'linear')
    plt.show()



# ============================================================================
# PYTORCH MLP MODEL
# ============================================================================

class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for binary classification."""
    
    # Constructor
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.0, activation: str = 'relu'):
        # Initialize the MLP model  
        super(MLPClassifier, self).__init__()
        # Create layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            # Add hidden layer
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        # Output layer (binary classification)
        self.layers.append(nn.Linear(hidden_dims[-1], 1))
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU() # ReLU activation
        elif activation == 'tanh':
            self.activation = nn.Tanh() # Tanh activation
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid() # Sigmoid activation
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None    
        self.sigmoid = nn.Sigmoid() # Sigmoid for output layer
    
    # Forward pass
    def forward(self, x):
        # Through hidden layers, with activation and dropout
        for layer in self.layers[:-1]:
            x = layer(x) # Linear layer
            x = self.activation(x) # Activation
            if self.dropout:
                x = self.dropout(x) # Dropout
        
        # Output layer
        x = self.layers[-1](x) # Linear output layer
        x = self.sigmoid(x) # Sigmoid activation for binary output
        return x


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

# train for one epoch
def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train() # set model to training mode
    total_loss = 0 # initialize total loss
    correct = 0 # initialize correct predictions
    total = 0 # initialize total samples
    
    # iterate over batches
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device) # move to device
        
        # zero gradients
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze(-1) # forward pass
        loss = criterion(outputs, y_batch) # compute loss
        
        loss.backward() # backward pass
        optimizer.step() # update weights
        
        total_loss += loss.item() * X_batch.size(0) # accumulate loss
        predicted = (outputs > 0.5).float() # threshold predictions
        correct += (predicted == y_batch).sum().item() # count correct
        total += y_batch.size(0) # count total samples
    
    return total_loss / total, correct / total # return average loss and accuracy


# validate for one epoch
def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch).squeeze(-1)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
            predicted = (outputs > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, early_stopping_patience=10):
    """Train model with early stopping."""
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses, train_accs, val_accs


# ============================================================================
# GRID SEARCH FOR HYPERPARAMETERS
# ============================================================================

def grid_search_mlp(X_train, y_train, X_val, y_val, 
                    param_grid: Dict, dataset_idx: int, device):
    """Perform grid search for MLP hyperparameters."""
    
    # Prepare data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val.values if hasattr(y_val, 'values') else y_val)
    )
    
    best_score = 0
    best_params = None
    best_model = None
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    
    print(f"Testing {len(combinations)} parameter combinations...")
    
    for i, combination in enumerate(combinations):
        params = dict(zip(keys, combination))
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, 
                                  batch_size=params['batch_size'], 
                                  shuffle=True)
        val_loader = DataLoader(val_dataset, 
                               batch_size=params['batch_size'], 
                               shuffle=False)
        
        # Create model
        model = MLPClassifier(
            input_dim=X_train.shape[1],
            hidden_dims=params['hidden_dims'],
            dropout=params['dropout'],
            activation=params['activation']
        ).to(device)
        
        # Create optimizer
        if params['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        elif params['optimizer'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=params['lr'], 
                                 momentum=0.9)
        else:
            optimizer = optim.RMSprop(model.parameters(), lr=params['lr'])
        
        criterion = nn.BCELoss()
        
        # Train model
        _, _, _, val_accs = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=100, device=device, early_stopping_patience=15
        )
        
        final_val_acc = val_accs[-1] if val_accs else 0
        
        if final_val_acc > best_score:
            best_score = final_val_acc
            best_params = params
            best_model = model.state_dict().copy()
    
    return best_params, best_model, scaler


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def train_mlp_grid_search(X_train, y_train, X_test, y_test, dataset_idx: int = 1):
    """
    Train MLP with grid search and show results:
      - best params
      - test accuracy + classification report
      - confusion matrix heatmap
      - learning curve
    
    Returns:
        best_params (dict): Best hyperparameters found.
        test_acc (float): Test accuracy score.
        f1 (float): F1 score on test set.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Split train into train/val (80/20)
    split_idx = int(0.8 * len(X_train))
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
    
    # Define parameter grid based on dataset
    if dataset_idx == 1:
        param_grid = {
            'hidden_dims': [[64], [64, 32], [128, 64]],
            'lr': [0.001, 0.01],
            'batch_size': [16, 32],
            'dropout': [0.0, 0.2],
            'activation': ['relu', 'tanh'],
            'optimizer': ['adam']
        }
    elif dataset_idx == 2:
        param_grid = {
            'hidden_dims': [[128, 64], [64, 32, 16]],
            'lr': [0.001, 0.005],
            'batch_size': [16, 32],
            'dropout': [0.2, 0.3],
            'activation': ['relu'],
            'optimizer': ['adam']
        }
    else:  # dataset 3
        param_grid = {
            'hidden_dims': [[64], [64, 32]], 
            'lr': [0.01, 0.001],
            'batch_size': [16, 32],
            'dropout': [0.0, 0.1],
            'activation': ['relu', 'tanh'],
            'optimizer': ['adam', 'sgd']
        }
    
    # Grid search
    best_params, best_model_state, scaler = grid_search_mlp(
        X_tr, y_tr, X_val, y_val, param_grid, dataset_idx, device
    )
    
    print("Best params:", best_params)
    
    # Train final model with best params on full train set
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.FloatTensor(y_test.values if hasattr(y_test, 'values') else y_test)
    )
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=best_params['batch_size'], 
                              shuffle=True)
    test_loader = DataLoader(test_dataset, 
                             batch_size=best_params['batch_size'], 
                             shuffle=False)
    
    # Create and train final model
    final_model = MLPClassifier(
        input_dim=X_train.shape[1],
        hidden_dims=best_params['hidden_dims'],
        dropout=best_params['dropout'],
        activation=best_params['activation']
    ).to(device)
    
    if best_params['optimizer'] == 'adam':
        optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'])
    elif best_params['optimizer'] == 'sgd':
        optimizer = optim.SGD(final_model.parameters(), lr=best_params['lr'], 
                             momentum=0.9)
    else:
        optimizer = optim.RMSprop(final_model.parameters(), lr=best_params['lr'])
    
    criterion = nn.BCELoss()
    
    train_losses, val_losses, train_accs, val_accs = train_model(
        final_model, train_loader, test_loader, criterion, optimizer,
        num_epochs=150, device=device, early_stopping_patience=20
    )
    
    # Evaluate on test set
    final_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = final_model(X_batch).squeeze(-1)
            predicted = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(predicted)
            all_labels.extend(y_batch.numpy())
    
    test_acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f"\nDataset: Monk-{dataset_idx}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("Classification report:")
    print(classification_report(all_labels, all_preds))
    
    # Plot confusion matrix
    plot_confusion_matrix_heatmap(all_labels, all_preds, dataset_idx=dataset_idx)
    # plot learning curve
    plot_learning_curve_single(train_losses, val_losses, train_accs, val_accs, dataset_idx=dataset_idx)
    return best_params, test_acc, f1


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results = {}
    
    print("="*80)
    print("ANALISI MLP (PyTorch) SU TUTTI I DATASET MONKS")
    print("="*80)
    
    for n_monk in [1, 2, 3]:
        print(f"\n{'='*80}")
        print(f"PROCESSING MONK-{n_monk}")
        print(f"{'='*80}")
        
        x_train, y_train, x_test, y_test = load_monk_data(n_monk)
        
        baseline_acc, majority_class = calculate_majority_baseline(
            y_train, y_test
        )
        print(f"Majority Voting Baseline Accuracy: {baseline_acc:.4f} "
              f"(Class: {majority_class})")
        
        best_params, test_acc, f1 = train_mlp_grid_search(
            x_train, y_train, x_test, y_test, dataset_idx=n_monk
        )
        
        results[n_monk] = {
            'baseline_acc': baseline_acc,
            'majority_class': majority_class,
            'best_params': best_params,
            'test_acc': test_acc,
            'f1_score': f1
        }
    
    # Stampa riepilogo finale
    print("\n" + "="*80)
    print("RIEPILOGO RISULTATI FINALI")
    print("="*80)
    
    for n_monk in [1, 2, 3]:
        res = results[n_monk]
        print(f"\n--- MONK-{n_monk} ---")
        print(f"Baseline Accuracy (Majority Class): {res['baseline_acc']:.4f} "
              f"(Class: {res['majority_class']})")
        print(f"Best Parameters: {res['best_params']}")
        print(f"Test Accuracy: {res['test_acc']:.4f}")
        print(f"F1 Score: {res['f1_score']:.4f}")
    
    print("\n" + "="*80)
    print("CONFRONTO PRESTAZIONI")
    print("="*80)
    print(f"{'Dataset':<10} {'Baseline':<12} {'Test Acc':<12} {'F1 Score':<12}")
    print("-"*80)
    for n_monk in [1, 2, 3]:
        res = results[n_monk]
        print(f"MONK-{n_monk:<5} {res['baseline_acc']:>8.4f}    "
              f"{res['test_acc']:>8.4f}    {res['f1_score']:>8.4f}")