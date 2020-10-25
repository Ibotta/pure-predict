import warnings
import numpy as np

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris

from pure_sklearn.map import convert_estimator
from pure_sklearn.utils import shape, tosparse

METHODS = [
    "predict",
    "predict_proba",
    "predict_log_proba",
]


def test_decision_tree_clf():
    X, y = load_iris(return_X_y=True)
    X_ = X.tolist()
    X_sparse = tosparse(X_)
    for y_ in [y, (y == 0).astype(int), (y == 2).astype(int)]:
        for max_depth in [5, 10, None]:
            clf = DecisionTreeClassifier(max_depth=max_depth, random_state=5)
            clf.fit(X, y_)
            clf_ = convert_estimator(clf)

            for method in METHODS:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = getattr(clf, method)(X)
                scores_ = getattr(clf_, method)(X_)
                scores_sparse = getattr(clf_, method)(X_sparse)
                assert np.allclose(scores, scores_, equal_nan=True)
                assert np.allclose(scores, scores_sparse, equal_nan=True)


def test_decision_tree_reg():
    X, y = load_iris(return_X_y=True)
    X_ = X.tolist()
    X_sparse = tosparse(X_)
    for y_ in [(y == 0).astype(int), (y == 2).astype(int)]:
        for max_depth in [5, 10, None]:
            clf = DecisionTreeRegressor(max_depth=max_depth, random_state=5)
            clf.fit(X, y_)
            clf_ = convert_estimator(clf)

            for method in ["predict"]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    scores = getattr(clf, method)(X)
                scores_ = getattr(clf_, method)(X_)
                scores_sparse = getattr(clf_, method)(X_sparse)
                assert np.allclose(scores.shape, shape(scores_))
                assert np.allclose(scores, scores_, equal_nan=True)
                assert np.allclose(scores, scores_sparse, equal_nan=True)
