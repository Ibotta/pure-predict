import warnings
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.datasets import load_iris

from pure_sklearn.map import convert_estimator
from pure_sklearn.utils import shape

MAX_ITER = 1000
TOL = 1e-3
METHODS = [
    "decision_function",
    "predict",
    "predict_proba",
    "predict_log_proba",
    "_predict_proba_lr",
]
LOSSES = [
    "hinge",
    "log",
    "modified_huber",
    "squared_hinge",
    "perceptron",
    "squared_loss",
    "huber",
    "epsilon_insensitive",
    "squared_epsilon_insensitive",
]


def test_sgd():
    X, y = load_iris(return_X_y=True)
    X_ = X.tolist()
    for y_ in [y, (y == 0).astype(int), (y == 2).astype(int)]:
        for loss in LOSSES:
            for fit_intercept in [True, False]:
                clf = SGDClassifier(
                    fit_intercept=fit_intercept, max_iter=MAX_ITER, tol=TOL, loss=loss
                )
                clf.fit(X, y_)
                clf_ = convert_estimator(clf)

                for method in METHODS:
                    if hasattr(clf, method) and hasattr(clf_, method):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            scores = getattr(clf, method)(X)
                        scores_ = getattr(clf_, method)(X_)
                        assert np.allclose(scores.shape, shape(scores_))
                        assert np.allclose(scores, scores_, equal_nan=True)
