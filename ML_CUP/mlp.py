from sklearn.model_selection import train_test_split, KFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
import time
import os
import platform

from ml_cup_data_loader import (
    load_training_set,
    load_test_set,
    mee,
    write_blind_results
)

# Create results directory
os.makedirs("./results", exist_ok=True)

start_time = time.time()
start_timestamp = time.strftime("%Y%m%d_%H%M%S")

activation_functions = {
    'relu': nn.ReLU,
    'leaky_relu': lambda: nn.LeakyReLU(0.1),
    'gelu': nn.GELU,
}

optimizers_dict = {
    'sgd': lambda params, lr, wd, momentum: optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum),
    'adam': lambda params, lr, wd, momentum: optim.Adam(params, lr=lr, weight_decay=wd),
    'adamw': lambda params, lr, wd, momentum: optim.AdamW(params, lr=lr, weight_decay=wd),
}

class MLCupNN(nn.Module):
    def __init__(self, input_dimension, n_hidden_layers, num_units, dropout, activation):
        super().__init__()

        layers = []
        prev_dim = input_dimension
        act_fn = activation_functions[activation]

        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(prev_dim, num_units))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            prev_dim = num_units

        layers.append(nn.Linear(prev_dim, 4))
        self.net = nn.Sequential(*layers)
        self._init_weights(activation)

    def _init_weights(self, activation):
        nonlin = 'leaky_relu' if activation == 'leaky_relu' else 'relu'
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlin)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

def mee_loss(y_pred, y_true):
    return torch.norm(y_pred - y_true, dim=1).mean()

def train_model(model, optimizer, scheduler, train_dataloader, val_dataloader, scaler_y, max_epochs=500, patience=30, clip_grad=1.0, min_delta=0.001):

    best_vl_loss = float("inf")
    best_state = None
    flat_epochs = 0

    tr_curve_values = []
    vl_curve_values = []
    
    for epoch in range(max_epochs):

        # run model training
        model.train()

        for Xb, yb in train_dataloader:
            optimizer.zero_grad()
            preds = model(Xb)
            loss = mee_loss(preds, yb)
            loss.backward()

            # if clip gradient is provided, apply
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

        # compute loss
        model.eval()
        tr_loss = 0.0
        
        with torch.no_grad():
            for Xb, yb in train_dataloader:
                preds = model(Xb)
                preds_unscaled = scaler_y.inverse_transform(preds.cpu().numpy())
                yb_u = scaler_y.inverse_transform(yb.cpu().numpy())
                tr_loss += mee(yb_u, preds_unscaled) * len(Xb)

        tr_loss /= len(train_dataloader.dataset)
        tr_curve_values.append(tr_loss)

        # run validation
        vl_loss = 0.0

        with torch.no_grad():
            for Xb, yb in val_dataloader:
                preds = model(Xb)
                preds_unscaled = scaler_y.inverse_transform(preds.cpu().numpy())
                yb_u = scaler_y.inverse_transform(yb.cpu().numpy())
                vl_loss += mee(yb_u, preds_unscaled) * len(Xb)

        vl_loss /= len(val_dataloader.dataset)
        vl_curve_values.append(vl_loss)

        if scheduler is not None:
            scheduler.step(vl_loss)

        # early stopping
        if vl_loss < best_vl_loss - min_delta:
            best_vl_loss = vl_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            flat_epochs = 0
        else:
            flat_epochs += 1
            if flat_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return tr_curve_values, vl_curve_values, best_vl_loss


############################################

train_df = load_training_set()
blind_df = load_test_set()

X = train_df.iloc[:, :-4].values
y = train_df.iloc[:, -4:].values
X_blind = blind_df.values

# split in dev set and hold out
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_hidden_layers': [2, 3],
    'num_units': [16, 64, 96],
    'momentum': [0.9],
    'learning_rate': [0.001, 0.2],
    'weight_decay': [0.02],
    'dropout': [0.3, 0.5, 0.6],
    'activation': ['gelu'],
    'optimizer': ['sgd', 'adamw']
}

print(f"Total configurations to test: {len(list(ParameterGrid(param_grid)))}")

# grid search and k fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

best_cv_loss = float("inf")
best_config = None
all_results = []

grid_search_start = time.time()

for config in tqdm(list(ParameterGrid(param_grid)), desc="Grid Search"):

    fold_losses = []
    fold_train_losses = []
    config_start = time.time()

    for fold_idx, (tr_idx, val_idx) in enumerate(kfold.split(X_dev)):

        X_tr_raw  = X_dev[tr_idx]
        X_val_raw = X_dev[val_idx]
        y_tr_raw  = y_dev[tr_idx]
        y_val_raw = y_dev[val_idx]

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        # scale input data
        X_tr = scaler_X.fit_transform(X_tr_raw)
        X_val = scaler_X.transform(X_val_raw)

        y_tr = scaler_y.fit_transform(y_tr_raw)
        y_val = scaler_y.transform(y_val_raw)

        # convert datasets to tensor dataset used by pytirch
        train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

        train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=128)

        # init model with current config
        model = MLCupNN(
            X.shape[1],
            config['n_hidden_layers'],
            config['num_units'],
            config['dropout'],
            config['activation'])

        # initialize optimizer
        optimizer = optimizers_dict[config['optimizer']](
            model.parameters(),
            config['learning_rate'],
            config['weight_decay'],
            config['momentum'])

        # this scheduler is used to reduce learning rate on plateau and provide smoother training on plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        tr_curve, vl_curve, vl_loss = train_model(model, optimizer, scheduler, train_dl, val_dl, scaler_y, max_epochs=1500, patience=60, min_delta=0.01)

        fold_losses.append(vl_loss)
        fold_train_losses.append(tr_curve[-1])

    mean_loss = np.mean(fold_losses)
    mean_train_loss = np.mean(fold_train_losses)

    all_results.append({
        'config': config,
        'mean_val_mee': mean_loss,
        'mean_train_mee': mean_train_loss,
        'std_val_mee': np.std(fold_losses),
        'std_train_mee': np.std(fold_train_losses)
    })

    if mean_loss < best_cv_loss:
        best_cv_loss = mean_loss
        best_config = config

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_dev_scaled = scaler_X.fit_transform(X_dev)
y_dev_scaled = scaler_y.fit_transform(y_dev)

X_test_scaled = scaler_X.transform(X_test)
X_blind_scaled = scaler_X.transform(X_blind)

# split dataset for final retrain
X_tr, X_val, y_tr, y_val = train_test_split(X_dev_scaled, y_dev_scaled, test_size=0.2, random_state=42)

train_dl = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)), batch_size=128, shuffle=True)
val_dl = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=128)

print(f"Best model: {best_config}\n")

model = MLCupNN(
    X.shape[1],
    best_config['n_hidden_layers'],
    best_config['num_units'],
    best_config['dropout'],
    best_config['activation'])

optimizer = optimizers_dict[best_config['optimizer']](
    model.parameters(),
    best_config['learning_rate'],
    best_config['weight_decay'],
    best_config['momentum'])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)


tr_curve_values, vl_curve_values, _ = train_model(model, optimizer, scheduler, train_dl, val_dl, scaler_y,max_epochs=1500, patience=60, min_delta=0.01)

# evaluate final model
model.eval()
with torch.no_grad():
    # tr set
    y_tr_pred = scaler_y.inverse_transform(model(torch.tensor(X_tr, dtype=torch.float32)).numpy())
    y_tr_true = scaler_y.inverse_transform(y_tr)
    train_mee_final = mee(y_tr_true, y_tr_pred)
    
    # vl set
    y_val_pred = scaler_y.inverse_transform(model(torch.tensor(X_val, dtype=torch.float32)).numpy())
    y_val_true = scaler_y.inverse_transform(y_val)
    val_mee_final = mee(y_val_true, y_val_pred)
    
    # hold oout test set
    y_test_pred = scaler_y.inverse_transform(model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy())
    test_mee_final = mee(y_test, y_test_pred)

    # blind set computation
    y_blind = scaler_y.inverse_transform(model(torch.tensor(X_blind_scaled, dtype=torch.float32)).numpy())

print("aggregated metrics:")
print(f"Best CV Val MEE: {best_cv_loss:.6f}")
print(f"Final params: {best_config}")
print(f"Final train MEE: {train_mee_final:.6f}")
print(f"Final val MEE:   {val_mee_final:.6f}")
print(f"Final test MEE:  {test_mee_final:.6f}")

write_blind_results("Pytorch_MLP", y_blind)

############################################
# LEARNING CURVES
############################################

# Plot 1: Full learning curve
fig, ax = plt.subplots(figsize=(12, 7))
ax.plot(tr_curve_values, label="Train MEE", linewidth=2.5, color='#1f77b4', alpha=0.9)
ax.plot(vl_curve_values, label="Validation MEE", linewidth=2.5, color='#ff7f0e', alpha=0.9)

all_losses = tr_curve_values + vl_curve_values
min_loss = min(all_losses)
max_loss = max(all_losses)
margin = (max_loss - min_loss) * 0.1
ax.set_ylim(min_loss - margin, max_loss + margin)

ax.set_xlabel("Epoch", fontsize=13, fontweight='bold')
ax.set_ylabel("MEE (original scale)", fontsize=13, fontweight='bold')
ax.set_title("Learning Curve - Full View", fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig(f"./results/learning_curve_full_{start_timestamp}.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Full learning curve saved")

# Plot 2: Zoomed view (skip first 20% of epochs)
skip_epochs = max(1, len(tr_curve_values) // 5)

fig, ax = plt.subplots(figsize=(12, 7))
epochs_zoomed = range(skip_epochs, len(tr_curve_values))
ax.plot(epochs_zoomed, tr_curve_values[skip_epochs:], 
        label="Train MEE", linewidth=2.5, color='#1f77b4', alpha=0.9)
ax.plot(epochs_zoomed, vl_curve_values[skip_epochs:], 
        label="Validation MEE", linewidth=2.5, color='#ff7f0e', alpha=0.9)

later_losses = tr_curve_values[skip_epochs:] + vl_curve_values[skip_epochs:]
min_loss = min(later_losses)
max_loss = max(later_losses)
margin = (max_loss - min_loss) * 0.15
ax.set_ylim(min_loss - margin, max_loss + margin)

ax.set_xlabel("Epoch", fontsize=13, fontweight='bold')
ax.set_ylabel("MEE (original scale)", fontsize=13, fontweight='bold')
ax.set_title(f"Learning Curve - Convergence View (from epoch {skip_epochs})", 
            fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='upper right', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.tight_layout()
plt.savefig(f"./results/learning_curve_zoomed_{start_timestamp}.png", dpi=300, bbox_inches='tight')
plt.show()
print(f"✓ Zoomed learning curve saved")
