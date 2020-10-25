import warnings
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

from pure_sklearn.map import convert_estimator
from pure_sklearn.utils import shape

METHODS = ["predict", "predict_proba", "predict_log_proba"]


def test_random_forest():
    X, y = load_iris(return_X_y=True)
    X_ = X.tolist()
    for y_ in [y, (y == 0).astype(int), (y == 2).astype(int)]:
        for n_estimators in [1, 10]:
            for max_depth in [5, 10, None]:
                clf = RandomForestClassifier(
                    bootstrap=False,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=5,
                )
                clf.fit(X, y_)
                clf_ = convert_estimator(clf)

                for method in METHODS:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        scores = getattr(clf, method)(X)
                    scores_ = getattr(clf_, method)(X_)
                    assert np.allclose(scores.shape, shape(scores_))
                    assert np.allclose(scores, scores_, equal_nan=True)
