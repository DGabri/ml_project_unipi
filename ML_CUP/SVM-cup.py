import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, make_scorer
from warnings import filterwarnings
from collections import Counter
from ml_cup_data_loader import (
    load_training_set,
    load_test_set,
    euclidean_distance_score,
    mee_scorer,
    write_blind_results,
)
filterwarnings("ignore")

# load data
train_df = load_training_set()
blind_df = load_test_set()
X = train_df.iloc[:, :-4].values
y = train_df.iloc[:, -4:].values
X_blind = blind_df.values

# hyperparameter grid
param_grid = {
    'estimator__kernel': ['rbf'],
    'estimator__C': [1,2,3,4,5, 6, 8, 10, 12, 15],
    'estimator__gamma': [0.8, 1.0, 1.1,1.2,1.3,1.4, 1.5],
    'estimator__epsilon': [0.25, 0.3, 0.35, 0.4, 0.5,0.6,0.7]
}

# initialize nested k fold cv
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=123)

# storage for results
outer_results = []
best_params_list = []

# nested k-fold CV
for fold, (tr_idx, ts_idx) in enumerate(outer_cv.split(X), 1):
    
    # split data
    X_train, X_test = X[tr_idx], X[ts_idx]
    y_train, y_test = y[tr_idx], y[ts_idx]
    
    # scaling 
    feat_scaler = StandardScaler()
    X_train_sc = feat_scaler.fit_transform(X_train)
    X_test_sc = feat_scaler.transform(X_test)
    target_scaler = StandardScaler()
    y_train_sc = target_scaler.fit_transform(y_train)
    
    # custom scorer in original scale
    def mee_unscaled(y_train_sc, y_pred_sc):
        y_train = target_scaler.inverse_transform(y_train_sc)
        y_pred = target_scaler.inverse_transform(y_pred_sc)
        return mee_scorer(y_train, y_pred)
    # scorer based on unscaled MEE
    scorer = make_scorer(mee_unscaled, greater_is_better=True)
    
    # MODEL SELECTION
    # inner grid search
    svr = SVR()
    multi_svr = MultiOutputRegressor(svr)
    
    gs = GridSearchCV(
        multi_svr,
        param_grid,
        scoring=scorer,
        cv=inner_cv,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    
    gs.fit(X_train_sc, y_train_sc)
    
    best_p = gs.best_params_
    best_params_list.append(best_p)
    
    # inner validation score (negated MEE from GridSearch)
    inner_val_mee = -gs.best_score_
    
    # MODEL ASSESSMENT
    # train error on outer train
    best_mdl = gs.best_estimator_
    y_train_pred_sc = best_mdl.predict(X_train_sc)
    y_train_pred = target_scaler.inverse_transform(y_train_pred_sc)
    tr_err = euclidean_distance_score(y_train, y_train_pred)
    tr_r2 = r2_score(y_train, y_train_pred)
    
    # test error on outer test
    y_test_pred_sc = best_mdl.predict(X_test_sc)
    y_test_pred = target_scaler.inverse_transform(y_test_pred_sc)
    te_err = euclidean_distance_score(y_test, y_test_pred)
    te_r2 = r2_score(y_test, y_test_pred)
    
    outer_results.append({'fold': fold,'train_mee': tr_err,'train_r2': tr_r2,'val_mee': inner_val_mee,'test_mee': te_err,'test_r2': te_r2,'params': best_p})
    print(f"Fold {outer_results[-1]['fold']}: Train={outer_results[-1]['train_mee']:.6f}, Val={outer_results[-1]['val_mee']:.6f}, "f"Test={outer_results[-1]['test_mee']:.6f}, R²={outer_results[-1]['test_r2']:.6f}")

# aggregate results
train_mees = [r['train_mee'] for r in outer_results]
val_mees = [r['val_mee'] for r in outer_results]
test_mees = [r['test_mee'] for r in outer_results]
test_r2s = [r['test_r2'] for r in outer_results]
print("aggregated metrics:")
print(f"Mean Train MEE:  {np.mean(train_mees):.6f} ± {np.std(train_mees):.6f}")
print(f"Mean Val MEE:    {np.mean(val_mees):.6f} ± {np.std(val_mees):.6f}")
print(f"Mean Test MEE:   {np.mean(test_mees):.6f} ± {np.std(test_mees):.6f}")
print(f"Mean Test R²:    {np.mean(test_r2s):.6f} ± {np.std(test_r2s):.6f}")

# analyze best params
C_vals = [p['estimator__C'] for p in best_params_list]
gamma_vals = [p['estimator__gamma'] for p in best_params_list]
epsilon_vals = [p['estimator__epsilon'] for p in best_params_list]

final_params = {
    'estimator__kernel': 'rbf',
    'estimator__C': float(np.median(C_vals)),
    'estimator__gamma': float(np.median(gamma_vals)),
    'estimator__epsilon': float(np.median(epsilon_vals))
}

print(f"Final params: C={final_params['estimator__C']}, "
      f"gamma={final_params['estimator__gamma']}, "
      f"epsilon={final_params['estimator__epsilon']}")


# retrain on full data
final_feat_sc = StandardScaler()
X_sc = final_feat_sc.fit_transform(X)
X_blind_sc = final_feat_sc.transform(X_blind)
final_tgt_sc = StandardScaler()
y_sc = final_tgt_sc.fit_transform(y)

final_svr = SVR(
    kernel='rbf',
    C=float(np.median(C_vals)),
    gamma=float(np.median(gamma_vals)),
    epsilon=float(np.median(epsilon_vals))
)

final_mdl = MultiOutputRegressor(final_svr)
final_mdl.fit(X_sc, y_sc)
y_pred_sc = final_mdl.predict(X_sc)
y_pred = final_tgt_sc.inverse_transform(y_pred_sc)
final_tr_mee = euclidean_distance_score(y, y_pred)
final_tr_r2 = r2_score(y, y_pred)
print(f"Final train MEE: {final_tr_mee:.6f}")
print(f"Final train R²:  {final_tr_r2:.6f}")

# Blind predictions
y_blind_sc = final_mdl.predict(X_blind_sc)
y_blind = final_tgt_sc.inverse_transform(y_blind_sc)

# write blind results
write_blind_results("", y_blind)