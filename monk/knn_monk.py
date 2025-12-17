
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from monks_data_loader import load_monk_data
from monk_utils import *
import numpy as np

monk_ids = [1, 2, 3]
best_params = {}

for monk_id in monk_ids:
    print(f"Monk dataset id: {monk_id}")
    X_train, y_train, X_test, y_test = load_monk_data(monk_id)

    knn = KNeighborsClassifier()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    params_grid = {
        "n_neighbors": range(1, 12),
        "weights": ["uniform", "distance"],
        "algorithm": ["brute", "kd_tree", "ball_tree"],
        "metric": ["minkowski", "euclidean", "manhattan"]
    }

    random_search = RandomizedSearchCV(
        estimator=knn,
        param_distributions=params_grid,
        n_iter=100,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=True
    )

    random_search.fit(X_train, y_train)
    
    baseline_accuracy, most_freq_class = calculate_majority_baseline(y_train, y_test)
    print(f"Baseline accuracy: {baseline_accuracy:.2f} (class: {most_freq_class})")

    best_params_run = random_search.best_params_
    best_cv_score = random_search.best_score_
    print("")
    print(f"Best parameters: {best_params_run}")
    print(f"Best CV accuracy: {(best_cv_score*100):.2f} %")

    best_model = random_search.best_estimator_

    # run on test set
    y_pred = best_model.predict(X_test)
    y_pred_train = best_model.predict(X_train)

    tr_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred)
    cv_mean = random_search.best_score_
    
    print("")
    print(f"TR accuracy: {(tr_acc*100):.2f} %")
    print(f"Test accuracy: {(test_acc*100):.2f} %")
    print(f"Train Validation delta: {((tr_acc - test_acc)*100):.2f} %")
    print(f"CV Mean: {(cv_mean*100):.2f} %")
    print("")
    best_params_run["tr_acc"] = tr_acc
    best_params_run["test_acc"] = test_acc
    best_params_run["cv_mean"] = cv_mean
    
    best_params[monk_id] = best_params_run

    print(classification_report(y_test, y_pred))
    print("====================================")