import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.datasets import load_iris

from pure_sklearn.map import convert_estimator
from pure_sklearn.utils import shape

METHODS = [
    "predict",
    "predict_proba",
    "predict_log_proba",
]


def test_import():
    from pure_sklearn import dummy

    assert True


def test_dummy():
    X, y = load_iris(return_X_y=True)
    X_ = X.tolist()
    for y_ in [y, (y == 0).astype(int), (y == 2).astype(int)]:
        clf = DummyClassifier(strategy="prior")
        clf.fit(X, y_)
        clf_ = convert_estimator(clf)

        for method in METHODS:
            scores = getattr(clf, method)(X)
            scores_ = getattr(clf_, method)(X_)
            assert np.allclose(scores.shape, shape(scores_))
            assert np.allclose(scores, scores_, equal_nan=True)
