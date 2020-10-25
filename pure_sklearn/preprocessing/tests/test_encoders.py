import numpy as np

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from pure_sklearn.map import convert_estimator


def test_onehotencoder():
    X0 = [["Male", 1], ["Female", 3], ["Female", 2]]
    X1 = [["Male", 1], ["Female", 27], ["Bananas", 2]]
    for X in [X0, X1]:
        ohe = OneHotEncoder(handle_unknown="ignore")
        ohe.fit(X)
        ohe_ = convert_estimator(ohe)
        assert np.allclose(ohe.transform(X).toarray(), ohe_.transform(X).todense())


def test_ordinalencoder():
    X0 = [["Male", 1], ["Female", 3], ["Female", 2]]
    X1 = [["Male", 1], ["Female", 27], ["Bananas", 2]]
    for X in [X0, X1]:
        ohe = OrdinalEncoder()
        ohe.fit(X)
        ohe_ = convert_estimator(ohe)
        assert np.allclose(ohe.transform(X), ohe_.transform(X))
