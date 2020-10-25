import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

from pure_sklearn.map import convert_estimator
from pure_sklearn.utils import shape, tosparse

SOLVER = "lbfgs"
MAX_ITER = 1000
METHODS = [
    "decision_function",
    "predict",
    "predict_proba",
    "predict_log_proba",
    "_predict_proba_lr",
]


def test_logistic():
    X, y = load_iris(return_X_y=True)
    X_ = X.tolist()
    X_sparse = tosparse(X_)
    for y_ in [y, (y == 0).astype(int), (y == 2).astype(int)]:
        for multi_class in ["ovr", "multinomial"]:
            for fit_intercept in [True, False]:
                clf = LogisticRegression(
                    solver=SOLVER,
                    multi_class=multi_class,
                    fit_intercept=fit_intercept,
                    max_iter=MAX_ITER,
                )
                clf.fit(X, y_)
                clf_ = convert_estimator(clf)

                for method in METHODS:
                    scores = getattr(clf, method)(X)
                    scores_ = getattr(clf_, method)(X_)
                    scores_sparse = getattr(clf_, method)(X_sparse)
                    assert np.allclose(scores, scores_)
                    assert np.allclose(scores, scores_sparse, equal_nan=True)
