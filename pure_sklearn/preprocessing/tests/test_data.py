import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    Normalizer,
    normalize,
)
from pure_sklearn.preprocessing import normalize_pure
from pure_sklearn.utils import tosparse
from pure_sklearn.map import convert_estimator

X = [[0.1, 1.0, 5.0], [3.3, 4.5, -0.2]]


def test_normalize():
    for norm in ["l1", "l2", "max"]:
        for axis in [0, 1]:
            assert np.allclose(
                normalize_pure(X, norm=norm, axis=axis),
                normalize(X, norm=norm, axis=axis),
            )


def test_normalize_sparse():
    X_sparse = tosparse(X)
    for norm in ["l1", "l2", "max"]:
        assert np.allclose(
            normalize_pure(X_sparse, norm=norm, axis=1).todense(),
            normalize(X, norm=norm, axis=1),
        )


def test_normalizer():
    for norm in ["l1", "l2", "max"]:
        tform = Normalizer(norm=norm)
        tform.fit(X)
        tform_ = convert_estimator(tform)
        X_t = tform.transform(X)
        X_t_ = tform_.transform(X)
        np.allclose(X_t, X_t_)


def test_normalizer_sparse():
    X_sparse = tosparse(X)
    for norm in ["l1", "l2", "max"]:
        tform = Normalizer(norm=norm)
        tform.fit(X)
        tform_ = convert_estimator(tform)
        X_t = tform.transform(X)
        X_t_ = tform_.transform(X_sparse)
        np.allclose(X_t, X_t_.todense())


def test_standard_scaler():
    for with_mean in [True, False]:
        for with_std in [True, False]:
            tform = StandardScaler(with_mean=with_mean, with_std=with_std)
            tform.fit(X)
            tform_ = convert_estimator(tform)
            X_t = tform.transform(X)
            X_t_ = tform_.transform(X)
            np.allclose(X_t, X_t_)


def test_standard_scaler_sparse():
    X_sparse = tosparse(X)
    for with_std in [True, False]:
        tform = StandardScaler(with_mean=False, with_std=with_std)
        tform.fit(X)
        tform_ = convert_estimator(tform)
        X_t = tform.transform(X)
        X_t_ = tform_.transform(X_sparse)
        np.allclose(X_t, X_t_.todense())


def test_max_abs_scaler():
    tform = MaxAbsScaler()
    tform.fit(X)
    tform_ = convert_estimator(tform)
    X_t = tform.transform(X)
    X_t_ = tform_.transform(X)
    np.allclose(X_t, X_t_)


def test_max_abs_scaler_sparse():
    X_sparse = tosparse(X)
    tform = MaxAbsScaler()
    tform.fit(X)
    tform_ = convert_estimator(tform)
    X_t = tform.transform(X)
    X_t_ = tform_.transform(X_sparse)
    np.allclose(X_t, X_t_.todense())


def test_min_max_scaler():
    for feature_range in [(0, 1), (1, 2), (-1, 1)]:
        tform = MinMaxScaler(feature_range=feature_range)
        tform.fit(X)
        tform_ = convert_estimator(tform)
        X_t = tform.transform(X)
        X_t_ = tform_.transform(X)
        np.allclose(X_t, X_t_)
