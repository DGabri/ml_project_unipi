
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

from ml_cup_data_loader import (
    load_training_set,
    load_test_set,
    mee,
    write_blind_results
)

class MLCupNN(nn.Module):
    def __init__(self, input_dimension, n_hidden_layers, num_units):
        super().__init__()

        layers = []
        prev_dim = input_dimension

        # create hidden layers
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(prev_dim, num_units))
            layers.append(nn.ReLU())
            prev_dim = num_units

        # add output layer of 4 units as we predict 4 outputs
        layers.append(nn.Linear(prev_dim, 4))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def mee_loss(y_pred, y_true):
    # euclidean distance for each point and compute mean at the end
    return torch.norm(y_pred - y_true, dim=1).mean()

def train_model(model, optimizer, train_dataloader, validation_dataloader, max_epochs=500, patience=30):
    
    best_validation_loss = float("inf")
    best_model_config = None
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
            optimizer.step()

            epoch_train_loss += loss.item() * len(Xb)

        # divide by the batch len to get mean error per batch
        epoch_train_loss /= len(train_dataloader.dataset)
        train_loss_list.append(epoch_train_loss)

        # validation loss calculation
        model.eval()
        validation_loss = 0.0

        with torch.no_grad():
            for Xb, yb in validation_dataloader:
                
                preds = model(Xb)
                loss = mee_loss(preds, yb)
                validation_loss += loss.item() * len(Xb)

        validation_loss /= len(validation_dataloader.dataset)
        validation_loss_list.append(validation_loss)

        # update best model state config
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_model_config = model.state_dict()
            num_epochs_flat_loss = 0
        else:
            num_epochs_flat_loss += 1
            
        if num_epochs_flat_loss >= patience: 
            break

    model.load_state_dict(best_model_config)
    return train_loss_list, validation_loss_list, best_validation_loss

########################################################################################################
# load datasets 
train_df = load_training_set()
blind_df = load_test_set()

# extract X and y from dataset
X_blind = blind_df.values
X = train_df.iloc[:, :-4].values
y = train_df.iloc[:, -4:].values

# 80% train - 20% test
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_dev_scaled = scaler_X.fit_transform(X_dev)
X_test_scaled = scaler_X.transform(X_test)
X_blind_scaled = scaler_X.transform(X_blind)

y_dev_scaled = scaler_y.fit_transform(y_dev)

# grid search setup
param_grid = {
    'n_hidden_layers': [1, 2],
    'num_units': [5, 10, 15, 20, 30],
    'momentum': [0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
}

# L2 regularization
weight_decay = 1e-4 
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

best_model_config = None
best_cv_loss = float("inf")

# run grid search with cross validation
params_to_test = list(ParameterGrid(param_grid))

for config in tqdm(params_to_test, desc="Running grid search"):

    fold_losses_list = []

    for train_idx, val_idx in kfold.split(X_dev_scaled):

        # divide in folds
        X_tr  = X_dev_scaled[train_idx]
        X_val = X_dev_scaled[val_idx]
        
        y_tr  = y_dev_scaled[train_idx]
        y_val = y_dev_scaled[val_idx]

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

        model = MLCupNN(
            input_dimension=X.shape[1],
            n_hidden_layers=config['n_hidden_layers'],
            num_units=config['num_units']
        )

        # SGD optimizer with momentum + L2 regularization
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=weight_decay
        )

        # train model
        train_loss_list, validation_loss_list, validation_loss = train_model(
            model,
            optimizer,
            train_dataloader,
            validation_dataloader,
        )

        fold_losses_list.append(validation_loss)

    avg_val_loss = np.mean(fold_losses_list)

    print(f"CV MEE: {avg_val_loss:.4f}. Best config: {config}")

    if avg_val_loss < best_cv_loss:
        best_cv_loss = avg_val_loss
        best_model_config = config

######################################

# retrain best model on full set
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

best_model = MLCupNN(
    input_dimension=X.shape[1],
    n_hidden_layers=best_model_config['n_hidden_layers'],
    num_units=best_model_config['num_units']
)

optimizer = optim.SGD(
    best_model.parameters(),
    lr=best_model_config['learning_rate'],
    momentum=best_model_config['momentum'],
    weight_decay=weight_decay
)

train_loss_list, validation_loss_list, _ = train_model(
    best_model,
    optimizer,
    train_dataloader,
    validation_dataloader
)

best_model.eval()
with torch.no_grad():
    y_test_pred_s = best_model(
        torch.tensor(X_test_scaled, dtype=torch.float32)
    ).cpu().numpy()

y_test_pred = scaler_y.inverse_transform(y_test_pred_s)

test_mee = mee(y_test, y_test_pred)

# blind prediction
with torch.no_grad():
    y_blind_s = best_model(
        torch.tensor(X_blind_scaled, dtype=torch.float32)
    ).cpu().numpy()

y_blind = scaler_y.inverse_transform(y_blind_s)
write_blind_results("Pytorch_MLP", y_blind)

print("Best configuration:", best_model_config)
print(f"CV Validation MEE: {best_cv_loss:.4f}")
print(f"Test MEE: {test_mee:.4f}")


# save figure to disk
plt.figure(figsize=(8,5))
plt.plot(train_loss_list, label='TR MEE')
plt.plot(validation_loss_list, label='VL MEE')
plt.xlabel('Epochs')
plt.ylabel('MEE')
plt.title(f"Learning Curve - Best Model (lr={best_model_config['learning_rate']})")
plt.legend()
plt.grid(True)


plt.savefig("./learning_curve_mlp.png", dpi=300)
plt.show()