from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
import time

from ml_cup_data_loader import (
    load_training_set,
    load_test_set,
    mee,
    write_blind_results
)

# activation functions dict
activation_functions = {
    'relu': nn.ReLU,
    'leaky_relu': lambda: nn.LeakyReLU(0.1),
    'gelu': nn.GELU,
}

# optimizers to test
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
        
        act_fn = activation_functions.get(activation, nn.ReLU)

        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(prev_dim, num_units))
            layers.append(nn.BatchNorm1d(num_units))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            prev_dim = num_units

        layers.append(nn.Linear(prev_dim, 4))
        self.net = nn.Sequential(*layers)
        
        self._init_weights(activation)
    
    def _init_weights(self, activation):
        
        if activation == 'leaky_relu':
            nonlin = 'leaky_relu' 
        else:
            nonlin = 'relu'
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity=nonlin)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)

def mee_loss(y_pred, y_true):
    return torch.norm(y_pred - y_true, dim=1).mean()

def train_model(model, optimizer, scheduler, train_dataloader, validation_dataloader, max_epochs=500, patience=30, clip_grad=1.0):
    
    best_validation_loss = float("inf")
    best_model_state = None
    num_epochs_flat_loss = 0

    train_loss_list = []
    validation_loss_list = []

    for epoch in range(max_epochs):

        model.train()
        epoch_train_loss = 0.0

        for Xb, yb in train_dataloader:
            optimizer.zero_grad()
            preds = model(Xb)
            loss = mee_loss(preds, yb)
            loss.backward()
            
            # prevent gradient explosion, keep training stable
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            
            optimizer.step()
            epoch_train_loss += loss.item() * len(Xb)

        epoch_train_loss /= len(train_dataloader.dataset)
        train_loss_list.append(epoch_train_loss)

        # validation
        model.eval()
        validation_loss = 0.0

        with torch.no_grad():
            for Xb, yb in validation_dataloader:
                preds = model(Xb)
                loss = mee_loss(preds, yb)
                validation_loss += loss.item() * len(Xb)

        validation_loss /= len(validation_dataloader.dataset)
        validation_loss_list.append(validation_loss)

        # scheduler
        if scheduler is not None:
            scheduler.step(validation_loss)

        # Early stopping
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model_state = model.state_dict()
            num_epochs_flat_loss = 0
        else:
            num_epochs_flat_loss += 1
            
        if num_epochs_flat_loss >= patience: 
            break

    model.load_state_dict(best_model_state)
    return train_loss_list, validation_loss_list, best_validation_loss

########################################################################################################

# load datasets
train_df = load_training_set()
blind_df = load_test_set()

X_blind = blind_df.values
X = train_df.iloc[:, :-4].values
y = train_df.iloc[:, -4:].values

X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_dev_scaled = scaler_X.fit_transform(X_dev)
X_test_scaled = scaler_X.transform(X_test)
X_blind_scaled = scaler_X.transform(X_blind)

y_dev_scaled = scaler_y.fit_transform(y_dev)

# grid search to test
param_grid = {
    'n_hidden_layers': [1, 2, 3],
    'num_units': [16, 24, 32, 64, 128],
    'momentum': [0.9],
    'learning_rate': [0.001, 0.005, 0.01, 0.05],
    'weight_decay': [1e-4, 1e-3],
    'dropout': [0.1, 0.2],
    'activation': ['relu', 'leaky_relu', 'gelu'],
    'optimizer': ['sgd', 'adam', 'adamw']
}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

best_model_config = None
best_cv_loss = float("inf")

# save all kfold values
best_fold_train = []
best_fold_val = []

params_to_test = list(ParameterGrid(param_grid))
print(f"Total configurations: {len(params_to_test)}")

for config in tqdm(params_to_test, desc="Running grid search"):

    fold_losses_list = []
    fold_train_curves = []
    fold_val_curves = []

    for train_idx, val_idx in kfold.split(X_dev_scaled):

        # get split training and validation set
        X_tr = X_dev_scaled[train_idx]
        X_val = X_dev_scaled[val_idx]
        y_tr = y_dev_scaled[train_idx]
        y_val = y_dev_scaled[val_idx]

        # convert to pytorch dataset
        train_dataset = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32)
        )
        validation_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )

        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=128, shuffle=False)

        # initialize model for current split
        model = MLCupNN(
            input_dimension=X.shape[1],
            n_hidden_layers=config['n_hidden_layers'],
            num_units=config['num_units'],
            dropout=config['dropout'],
            activation=config['activation']
        )

        # get optimizer to test
        optimizer = optimizers_dict[config['optimizer']](
            model.parameters(),
            lr=config['learning_rate'],
            wd=config['weight_decay'],
            momentum=config['momentum']
        )
        
        # this automatically reduces learning rate on plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=1e-6
        )

        # execute training 
        train_loss_list, validation_loss_list, validation_loss = train_model(
            model,
            optimizer,
            scheduler,
            train_dataloader,
            validation_dataloader,
        )

        fold_losses_list.append(validation_loss)
        fold_train_curves.append(train_loss_list)
        fold_val_curves.append(validation_loss_list)

    avg_val_loss = np.mean(fold_losses_list)

    if avg_val_loss < best_cv_loss:
        best_cv_loss = avg_val_loss
        best_model_config = config
        best_fold_train = fold_train_curves
        best_fold_val = fold_val_curves
        print(f"\nCV MEE: {avg_val_loss:.4f} Current model: {config}\n")

######################################
# retrain best model on training set

X_tr, X_val, y_tr, y_val = train_test_split(
    X_dev_scaled, y_dev_scaled, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(
    torch.tensor(X_tr, dtype=torch.float32),
    torch.tensor(y_tr, dtype=torch.float32)
)
validation_dataset = TensorDataset(
    torch.tensor(X_val, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32)
)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=128, shuffle=False)

best_model_config = None

# best run
# {'activation': 'relu', 'dropout': 0.1, 'learning_rate': 0.05, 'momentum': 0.9, 'n_hidden_layers': 3, 'num_units': 128, 'optimizer': 'adamw', 'weight_decay': 0.001}

best_model = MLCupNN(
    input_dimension=X.shape[1],
    n_hidden_layers=best_model_config['n_hidden_layers'],
    num_units=best_model_config['num_units'],
    dropout=best_model_config['dropout'],
    activation=best_model_config['activation']
)

optimizer = optimizers_dict[best_model_config['optimizer']](
    best_model.parameters(),
    lr=best_model_config['learning_rate'],
    wd=best_model_config['weight_decay'],
    momentum=best_model_config['momentum']
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=10,
    min_lr=1e-6
)

train_loss_list, validation_loss_list, _ = train_model(
    best_model,
    optimizer,
    scheduler,
    train_dataloader,
    validation_dataloader
)

# test evaluation
best_model.eval()
with torch.no_grad():
    y_test_pred_s = best_model(
        torch.tensor(X_test_scaled, dtype=torch.float32)
    ).cpu().numpy()

y_test_pred = scaler_y.inverse_transform(y_test_pred_s)
test_mee = mee(y_test, y_test_pred)

# run blind prediction
with torch.no_grad():
    y_blind_s = best_model(
        torch.tensor(X_blind_scaled, dtype=torch.float32)
    ).cpu().numpy()

y_blind = scaler_y.inverse_transform(y_blind_s)
write_blind_results("Pytorch_MLP", y_blind)

print("Best configuration:", best_model_config)
print(f"CV Validation MEE: {best_cv_loss:.4f}")
print(f"Test MEE: {test_mee:.4f}")

######################################
# plot
start = int(time.time())

# learning curve
plt.figure(figsize=(8, 5))
plt.plot(train_loss_list, label='TR MEE')
plt.plot(validation_loss_list, label='VL MEE')
plt.xlabel('Num Epochs')
plt.ylabel('MEE')
plt.title("Learning Curve")
plt.legend()
plt.grid(True)
plt.savefig(f"./results/learning_curve_mlp_{start}.png", dpi=300)
plt.show()