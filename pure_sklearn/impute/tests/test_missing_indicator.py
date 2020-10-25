import numpy as np

from sklearn.impute import MissingIndicator
from sklearn.datasets import load_iris

from pure_sklearn.map import convert_estimator
from pure_sklearn.utils import shape


def test_missing_indicator():
    X, y = load_iris(return_X_y=True)
    for missing_values in [np.nan, X[0][0], X[-1][1]]:
        X, y = load_iris(return_X_y=True)
        if np.isnan(missing_values):
            X.ravel()[np.random.choice(X.size, 20, replace=False)] = np.nan
        X_ = X.tolist()
        for features in ["missing-only", "all"]:
            imp = MissingIndicator(
                features=features, missing_values=missing_values, error_on_new=False
            )
            imp.fit(X)
            imp_ = convert_estimator(imp)

            X_t = getattr(imp, "transform")(X)
            X_t_ = getattr(imp_, "transform")(X_)
            assert np.allclose(X_t.shape, shape(X_t_))
            assert np.allclose(X_t, X_t_)
