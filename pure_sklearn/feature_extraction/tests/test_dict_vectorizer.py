import numpy as np

from sklearn.feature_extraction import DictVectorizer
from pure_sklearn.map import convert_estimator

X = [{"pizza": 1, "tacos": 2}, {}, {"bananas": 3}, {"tacos": 3}]


def test_dict_vectorizer():
    dv = DictVectorizer()
    dv.fit(X)
    dv_ = convert_estimator(dv)
    dv_t = dv.transform(X)
    dv_t_ = dv_.transform(X)
    assert np.allclose(dv_t.toarray(), dv_t_.todense())


def test_dict_vectorizer_dense():
    dv = DictVectorizer(sparse=False)
    dv.fit(X)
    dv_ = convert_estimator(dv)
    dv_t = dv.transform(X)
    dv_t_ = dv_.transform(X)
    assert np.allclose(dv_t, dv_t_)
