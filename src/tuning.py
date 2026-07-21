# src/tuning.py
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from src.models import MODEL_REGISTRY

RANDOM_STATE = 42

PARAM_GRIDS = {
    "logreg": {"C": [0.01, 0.1, 1, 10, 100], "solver": ["lbfgs"]},
    "decision_tree": {"max_depth": [3, 5, 7, 10, None],
                       "min_samples_split": [2, 5, 10],
                       "min_samples_leaf": [1, 2, 4]},
    "random_forest": {"n_estimators": [100, 200, 300],
                       "max_depth": [5, 10, 15, None],
                       "min_samples_split": [2, 5, 10],
                       "min_samples_leaf": [1, 2, 4]},
    "knn": {"n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"]},
    "svm": {"C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto", 0.01, 0.1],
            "kernel": ["rbf", "linear"]},
    "xgboost": {"n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0]},
    "lightgbm": {"n_estimators": [100, 200, 300],
                 "max_depth": [3, 5, 7, -1],
                 "learning_rate": [0.01, 0.05, 0.1],
                 "num_leaves": [15, 31, 63]},
}


def tune_model(name, X_train, y_train, scoring="recall",
               cv_splits=5, n_iter=20, random_state=RANDOM_STATE):
    """
    Randomized search over the registered param grid for `name`.
    Must be called on X_train/y_train ONLY (the already-resampled
    training set from Day 5) — never on X_test.
    """
    base_model = clone(MODEL_REGISTRY[name])
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        base_model, PARAM_GRIDS[name],
        n_iter=n_iter, scoring=scoring, cv=cv,
        random_state=random_state, n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train)
    return search