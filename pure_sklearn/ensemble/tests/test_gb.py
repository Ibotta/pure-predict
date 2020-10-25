import warnings
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris

from pure_sklearn.map import convert_estimator
from pure_sklearn.utils import shape

METHODS = ["predict", "predict_proba", "predict_log_proba"]


def test_gradient_boosting():
    X, y = load_iris(return_X_y=True)
    X_ = X.tolist()
    for y_ in [y, (y == 0).astype(int), (y == 2).astype(int)]:
        for n_estimators in [1, 10]:
            for max_depth in [5, 10, None]:
                for loss in ["exponential", "deviance"]:
                    if ((len(np.unique(y)) == 2) and loss == "exponential") or (
                        loss == "deviance"
                    ):
                        clf = GradientBoostingClassifier(
                            n_estimators=n_estimators,
                            random_state=5,
                            max_depth=max_depth,
                            loss=loss,
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
