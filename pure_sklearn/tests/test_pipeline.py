import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

from pure_sklearn.map import convert_estimator
from pure_sklearn.utils import tosparse, shape


def test_import():
    from pure_sklearn import pipeline

    assert True


def test_pipeline():
    X, y = load_iris(return_X_y=True)
    X_ = X.tolist()

    lr = LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=1000)
    pipe = Pipeline(
        steps=[
            ("imp", SimpleImputer()),
            ("lr", lr),
        ]
    )
    pipe.fit(X, y)
    pipe_ = convert_estimator(pipe)
    assert np.allclose(pipe.predict_proba(X), pipe_.predict_proba(X.tolist()))


def test_feature_union():
    X, y = load_iris(return_X_y=True)
    X_ = X.tolist()

    union = FeatureUnion(
        [
            ("imp_mean", SimpleImputer(strategy="mean")),
            ("imp_median", SimpleImputer(strategy="median")),
        ]
    )
    union.fit(X, y)
    union_ = convert_estimator(union)
    assert np.allclose(union.transform(X), union_.transform(X.tolist()))


def test_feature_union_sparse():
    X, y = load_iris(return_X_y=True)
    X_ = tosparse(X.tolist())

    union = FeatureUnion(
        [("ss", StandardScaler(with_mean=False)), ("mms", MaxAbsScaler())]
    )
    union.fit(X, y)
    union_ = convert_estimator(union)
    assert np.allclose(union.transform(X), union_.transform(X_).todense())
