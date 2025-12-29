import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, make_scorer
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
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

# custom scorer in original scale
def mee_unscaled(y_true_sc, y_pred_sc):
    y_true = target_scaler.inverse_transform(y_true_sc)
    y_pred = target_scaler.inverse_transform(y_pred_sc)
    return mee_scorer(y_true, y_pred)

# load data
train_df = load_training_set()
blind_df = load_test_set()
X = train_df.iloc[:, :-4].values
y = train_df.iloc[:, -4:].values
X_blind = blind_df.values

# hyperparameter grid
param_grid = {
    'estimator__n_estimators': [200, 300, 400],
    'estimator__learning_rate': [0.03, 0.05, 0.07],
    'estimator__max_depth': [5, 7, 9],
    'estimator__num_leaves': [31, 50, 70],
    'estimator__min_child_samples': [20, 30],
    'estimator__subsample': [0.8, 0.9],
    'estimator__colsample_bytree': [0.8, 0.9],
    'estimator__reg_alpha': [0.01, 0.1],
    'estimator__reg_lambda': [0.01, 0.1]
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
    
    # scorer based on unscaled MEE
    scorer = make_scorer(mee_unscaled, greater_is_better=True)
    
    # inner grid search
    lgbm = LGBMRegressor(
        random_state=42,
        verbose=-1,
        n_jobs=1,
        boosting_type='gbdt'
    )
    multi_lgbm = MultiOutputRegressor(lgbm, n_jobs=-1)
    
    gs = GridSearchCV(
        multi_lgbm,
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

# most common hyperparameters
param_keys = best_params_list[0].keys()
most_common_params = {}
for key in param_keys:
    values = [p[key] for p in best_params_list]
    most_common = Counter(values).most_common(1)[0][0]
    most_common_params[key] = most_common

print(f"Final params (most common): {most_common_params}")

# retrain on full data
final_feat_sc = StandardScaler()
X_sc = final_feat_sc.fit_transform(X)
X_blind_sc = final_feat_sc.transform(X_blind)
final_tgt_sc = StandardScaler()
y_sc = final_tgt_sc.fit_transform(y)

final_lgbm = LGBMRegressor(
    n_estimators=most_common_params['estimator__n_estimators'],
    learning_rate=most_common_params['estimator__learning_rate'],
    max_depth=most_common_params['estimator__max_depth'],
    num_leaves=most_common_params['estimator__num_leaves'],
    min_child_samples=most_common_params['estimator__min_child_samples'],
    subsample=most_common_params['estimator__subsample'],
    colsample_bytree=most_common_params['estimator__colsample_bytree'],
    reg_alpha=most_common_params['estimator__reg_alpha'],
    reg_lambda=most_common_params['estimator__reg_lambda'],
    random_state=42,
    verbose=-1,
    n_jobs=-1,
    boosting_type='gbdt'
)
final_mdl = MultiOutputRegressor(final_lgbm, n_jobs=-1)
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