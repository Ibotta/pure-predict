import numpy as np

from sklearn.feature_extraction.text import (
    HashingVectorizer,
    CountVectorizer,
    TfidfVectorizer,
)
from pure_sklearn.map import convert_estimator

X = [
    "how now brown cow",
    "brown cow bananas",
    "cheerios",
    "bananas are great",
    "great muffins are good",
    "the muffin's top is chocolate",
]


def test_count_vectorizer():
    vec = CountVectorizer()
    vec.fit(X)
    vec_ = convert_estimator(vec)
    assert np.allclose(vec.transform(X).toarray(), vec_.transform(X).todense())


def test_tfidf_vectorizer():
    for norm in ["l1", "l2", None]:
        vec = TfidfVectorizer(norm=norm)
        vec.fit(X)
        vec_ = convert_estimator(vec)
        assert np.allclose(vec.transform(X).toarray(), vec_.transform(X).todense())


def test_hashing_vectorizer():
    for norm in ["l1", "l2", None]:
        vec = HashingVectorizer(n_features=2 ** 8, norm=norm)
        vec.fit(X)
        vec_ = convert_estimator(vec)
        X_t = vec.transform(X)
        X_t_ = vec_.transform(X)
        assert np.allclose(vec.transform(X).toarray(), vec_.transform(X).todense())
