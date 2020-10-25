import numpy as np

from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris

from pure_sklearn.map import convert_estimator
from pure_sklearn.utils import shape

MAX_ITER = 1000
TOL = 1e-3
METHODS = ["decision_function", "predict", "_predict_proba_lr"]


def test_perceptron():
    X, y = load_iris(return_X_y=True)
    X_ = X.tolist()
    for y_ in [y, (y == 0).astype(int), (y == 2).astype(int)]:
        for fit_intercept in [True, False]:
            clf = Perceptron(fit_intercept=fit_intercept, max_iter=MAX_ITER, tol=TOL)
            clf.fit(X, y_)
            clf_ = convert_estimator(clf)

            for method in METHODS:
                scores = getattr(clf, method)(X)
                scores_ = getattr(clf_, method)(X_)
                assert np.allclose(scores.shape, shape(scores_))
                assert np.allclose(scores, scores_)
