import pytest
import sys
import numpy as np

try:
    import xgboost
    from xgboost import XGBClassifier
except ImportError:
    pass
from sklearn.datasets import load_iris

from pure_sklearn.map import convert_estimator

METHODS = ["predict", "predict_proba"]


@pytest.mark.skipif("xgboost" not in sys.modules, reason="requires xgboost")
def test_xgboost():
    X, y = load_iris(return_X_y=True)
    X_ = X.tolist()
    for y_ in [y, (y == 0).astype(int), (y == 2).astype(int)]:
        for n_estimators in [2, 10]:
            for max_depth in [3, 10]:
                clf = XGBClassifier(
                    booster="gbtree",
                    random_state=5,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                )
                clf.fit(X, y_)
                clf_ = convert_estimator(clf)
                for method in METHODS:
                    scores = getattr(clf, method)(X)
                    scores_ = getattr(clf_, method)(X_)
                    assert np.allclose(scores, scores_, equal_nan=True)
