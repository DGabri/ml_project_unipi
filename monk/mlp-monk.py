import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, ParameterGrid, StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from monks_data_loader import load_monk_data
from monk_utils import plot_confusion_matrix, calculate_majority_baseline,plot_curves,set_seed

# neural network model
class MLP(nn.Module):

    # constructor
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    # forward method
    def forward(self, x):
        return self.net(x).squeeze(1)
    # initialize weights 
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

# training function with early stopping
def train_net(model, opt, train_dl, val_dl, patience=15, max_ep=200, track=False):
    # initialize weights
    model.init_weights()
    criterion = nn.BCELoss()
    best_loss = float('inf')
    best_weights = None
    no_improve = 0
    
    # tracking
    tr_errs = []
    val_errs = []

    # training loop with early stopping
    for epoch in range(max_ep):
        model.train()
        tr_correct = 0
        tr_total = 0
        
        # training epoch
        for xb, yb in train_dl:
            opt.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            
            # tracking accuracy
            if track:
                pred = (out > 0.5).int()
                tr_correct += (pred == yb.int()).sum().item()
                tr_total += len(yb)

        # validation epoch
        model.eval()
        v_loss = 0
        v_correct = 0
        v_total = 0
        
        # validation loop
        with torch.no_grad():
            for xb, yb in val_dl:
                out = model(xb)
                loss = criterion(out, yb)
                v_loss += loss.item() * len(xb)
                # tracking accuracy
                if track:
                    pred = (out > 0.5).int()
                    v_correct += (pred == yb.int()).sum().item()
                    v_total += len(yb)
        
        # average validation loss
        v_loss /= len(val_dl.dataset)
        
        # tracking errors
        if track:
            tr_err = 1 - (tr_correct / tr_total)
            v_err = 1 - (v_correct / v_total)
            tr_errs.append(tr_err)
            val_errs.append(v_err)

        # early stopping check
        if v_loss < best_loss:
            best_loss = v_loss
            best_weights = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    
    if track:
        return model, tr_errs, val_errs
    return model

# evaluation function
def eval_model(model, dl):
    model.eval()
    correct = 0
    total = 0
    # evaluation loop
    with torch.no_grad():
        for xb, yb in dl:
            out = model(xb)
            pred = (out > 0.5).int()
            correct += (pred == yb.int()).sum().item()
            total += len(yb)
    return correct / total if total > 0 else 0

# hyperparameter grid
def get_params(dataset_num):
    if dataset_num == 1:
        return {
            'n_units': [3, 4],
            'lr': [0.5, 0.8],
            'bs': [1, 2],
            'mom': [0.8, 0.9],
            'wd': [0.0]
        }
    elif dataset_num == 2:
        return {
            'n_units': [3, 4],
            'lr': [0.2, 0.3],
            'bs': [1, 2],
            'mom': [0.5],
            'wd': [0.0, 0.001]
        }
    else:
        return {
            'n_units': [4, 5],
            'lr': [0.01, 0.05],
            'bs': [4, 8],
            'mom': [0.9, 0.92],
            'wd': [0.0005, 0.001, 0.005]
        }

# nested cross-validation
def nested_cv(X_tr, y_tr, dataset_num, inner_k=3, outer_k=5):
    
    # prepare data
    X = X_tr.values if hasattr(X_tr, 'values') else X_tr
    y = y_tr.values if hasattr(y_tr, 'values') else y_tr
    
    # get hyperparameter grid
    params = get_params(dataset_num)
    outer = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=42)
    
    scores = []
    best_params_list = []
    
    # MODEL ASSESSMENT
    # outer cross-validation
    for fold, (tr_idx, val_idx) in enumerate(outer.split(X, y), 1):
        
        X_tr_f = X[tr_idx]
        X_val_f = X[val_idx]
        y_tr_f = y[tr_idx]
        y_val_f = y[val_idx]
        
        # internal cross-validation for hyperparameter tuning
        inner = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=42)
        grid = list(ParameterGrid(params))
        
        best_cfg = None
        best_score = -np.inf
        
        # MODEL SELECTION
        # inner cross-validation
        for cfg in grid:
            inner_scores = []
            # inner folds
            for itr_idx, ival_idx in inner.split(X_tr_f, y_tr_f):
                X_itr = torch.FloatTensor(X_tr_f[itr_idx])
                y_itr = torch.FloatTensor(y_tr_f[itr_idx])
                X_ival = torch.FloatTensor(X_tr_f[ival_idx])
                y_ival = torch.FloatTensor(y_tr_f[ival_idx])
                
                # create dataloaders
                tr_ds = TensorDataset(X_itr, y_itr)
                val_ds = TensorDataset(X_ival, y_ival)
                
                tr_dl = DataLoader(tr_ds, batch_size=cfg['bs'], shuffle=True)
                val_dl = DataLoader(val_ds, batch_size=cfg['bs'], shuffle=False)

                # train model
                net = MLP(X.shape[1], cfg['n_units'])
                opt = optim.SGD(net.parameters(), lr=cfg['lr'], 
                               momentum=cfg['mom'], weight_decay=cfg['wd'])
                
                pat = 10 if dataset_num == 1 else 15
                maxe = 100 if dataset_num == 1 else 200
                
                # train and evaluate
                net = train_net(net, opt, tr_dl, val_dl, patience=pat, 
                              max_ep=maxe, track=False)
                sc = eval_model(net, val_dl)
                inner_scores.append(sc)
            
            avg_sc = np.mean(inner_scores)
            # check for best config
            if avg_sc > best_score:
                best_score = avg_sc
                best_cfg = cfg
        
        # final model training with best hyperparameters
        X_tr_t = torch.FloatTensor(X_tr_f)
        y_tr_t = torch.FloatTensor(y_tr_f)
        X_val_t = torch.FloatTensor(X_val_f)
        y_val_t = torch.FloatTensor(y_val_f)
        
        tr_ds = TensorDataset(X_tr_t, y_tr_t)
        val_ds = TensorDataset(X_val_t, y_val_t)
        
        tr_dl = DataLoader(tr_ds, batch_size=best_cfg['bs'], shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=best_cfg['bs'], shuffle=False)

        # train final model
        final_net = MLP(X.shape[1], best_cfg['n_units'])
        opt = optim.SGD(final_net.parameters(), lr=best_cfg['lr'],
                       momentum=best_cfg['mom'], weight_decay=best_cfg['wd'])
        
        pat = 10 if dataset_num == 1 else 15
        maxe = 100 if dataset_num == 1 else 200
        
        final_net = train_net(final_net, opt, tr_dl, val_dl, 
                            patience=pat, max_ep=maxe, track=False)
        fold_sc = eval_model(final_net, val_dl)
        
        scores.append(fold_sc)
        best_params_list.append(best_cfg)
        
        print(f"Fold {fold}/{outer_k} | params: {best_cfg} | inner score: {best_score:.4f} | outer score: {fold_sc:.4f}")
    
    scores = np.array(scores)
    
    print(f"\nNested CV results - Monk-{dataset_num}\nMean: {scores.mean():.4f} ± {scores.std():.4f}\nRange: [{scores.min():.4f}, {scores.max():.4f}]\n")
    
    return scores, best_params_list

# final training on full training set and testing
def train_final(X_tr, y_tr, X_te, y_te, dataset_num):
    
    X_tr = X_tr.values if hasattr(X_tr, 'values') else X_tr
    y_tr = y_tr.values if hasattr(y_tr, 'values') else y_tr
    X_te = X_te.values if hasattr(X_te, 'values') else X_te
    y_te = y_te.values if hasattr(y_te, 'values') else y_te
    
    # split train/val
    X_t, X_v, y_t, y_v = train_test_split(X_tr, y_tr, test_size=0.2, 
                                          random_state=42, stratify=y_tr)
    
    params = get_params(dataset_num)
    grid = list(ParameterGrid(params))
    
    best_cfg = None
    best_acc = 0
    best_net = None
    best_tr_errs = None
    best_val_errs = None

    # hyperparameter search
    for i, cfg in enumerate(grid):
        X_t_t = torch.FloatTensor(X_t)
        y_t_t = torch.FloatTensor(y_t)
        X_v_t = torch.FloatTensor(X_v)
        y_v_t = torch.FloatTensor(y_v)
        
        tr_ds = TensorDataset(X_t_t, y_t_t)
        val_ds = TensorDataset(X_v_t, y_v_t)
        
        tr_dl = DataLoader(tr_ds, batch_size=cfg['bs'], shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=cfg['bs'], shuffle=False)
        
        net = MLP(X_tr.shape[1], cfg['n_units'])
        opt = optim.SGD(net.parameters(), lr=cfg['lr'],
                       momentum=cfg['mom'], weight_decay=cfg['wd'])
        
        pat = 10 if dataset_num == 1 else 15
        maxe = 100 if dataset_num == 1 else 200
        
        net, tr_e, val_e = train_net(net, opt, tr_dl, val_dl,
                                    patience=pat, max_ep=maxe, track=True)
        acc = eval_model(net, val_dl)
        
        if acc > best_acc:
            best_acc = acc
            best_cfg = cfg
            best_net = net
            best_tr_errs = tr_e
            best_val_errs = val_e
        
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(grid)} done, best: {best_acc:.4f}")
    
    print(f"\nBest config: {best_cfg}")
    print(f"Val accuracy: {best_acc:.4f}")
    
    if best_tr_errs and best_val_errs:
        plot_curves(best_tr_errs, best_val_errs, dataset_num, " (best)")
    
    # test
    X_te_t = torch.FloatTensor(X_te)
    y_te_t = torch.FloatTensor(y_te)
    te_ds = TensorDataset(X_te_t, y_te_t)
    te_dl = DataLoader(te_ds, batch_size=32, shuffle=False)
    
    test_acc = eval_model(best_net, te_dl)
    
    best_net.eval()
    with torch.no_grad():
        preds = best_net(X_te_t).cpu().numpy()
        preds_bin = (preds > 0.5).astype(int)
    
    print(f"\nTest accuracy: {test_acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_te, preds_bin))
    
    return best_net, best_cfg, test_acc, preds_bin

# main analysis function
def run_analysis(X_tr, y_tr, X_te, y_te, dataset_num):
    # baseline
    base_acc, maj_class = calculate_majority_baseline(y_tr, y_te)
    print(f"Majority baseline: {base_acc:.4f} (class {maj_class})")
    
    # nested cv
    cv_scores, params_list = nested_cv(X_tr, y_tr, dataset_num, 
                                       inner_k=3, outer_k=5)
    
    # modello finale
    best_net, best_cfg, test_acc, preds = train_final(X_tr, y_tr, X_te, y_te, 
                                                       dataset_num)
    
    # visualizzazioni
    y_te = y_te.values if hasattr(y_te, 'values') else y_te
    plot_confusion_matrix(y_te, preds, dataset_num)
    return {
        'baseline': base_acc,
        'maj_class': maj_class,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'params_per_fold': params_list,
        'final_params': best_cfg,
        'test_acc': test_acc,
    }


if __name__ == "__main__":
    set_seed(42)
    
    results = {}
    
    for n in [1, 2, 3]:
        x_tr, y_tr, x_te, y_te = load_monk_data(n)
        print(f"\nMONK-{n} ANALYSIS")
        results[n] = run_analysis(x_tr, y_tr, x_te, y_te, n)