import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.datasets import load_iris

from pure_sklearn.map import convert_estimator
from pure_sklearn.utils import shape


def test_simple_imputer():
    X, y = load_iris(return_X_y=True)
    for missing_values in [np.nan, X[0][0], X[-1][1]]:
        X, y = load_iris(return_X_y=True)
        if np.isnan(missing_values):
            X.ravel()[np.random.choice(X.size, 20, replace=False)] = np.nan
        X_ = X.tolist()
        for strategy in ["mean", "median", "most_frequent", "constant"]:
            for add_indicator in [True, False]:
                imp = SimpleImputer(strategy=strategy, missing_values=missing_values)
                if hasattr(imp, "add_indicator"):
                    imp.add_indicator = add_indicator
                else:
                    imp.add_indicator = False
                imp.fit(X)
                imp_ = convert_estimator(imp)

                X_t = getattr(imp, "transform")(X)
                X_t_ = getattr(imp_, "transform")(X_)
                assert np.allclose(X_t.shape, shape(X_t_))
                assert np.allclose(X_t, X_t_)
