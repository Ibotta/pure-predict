import numpy as np

from sklearn.metrics.pairwise import cosine_similarity, cosine_distances, linear_kernel
from pure_sklearn.metrics.pairwise import (
    cosine_similarity_pure,
    cosine_distances_pure,
    linear_kernel_pure,
)

DIM1 = 10
DIM2 = 2


def test_cosine_similarity():
    X = np.random.rand(DIM1, DIM2)
    Y = np.random.rand(DIM1, DIM2)
    X_list = X.tolist()
    Y_list = Y.tolist()
    assert np.allclose(cosine_similarity(X, Y), cosine_similarity_pure(X_list, Y_list))


def test_cosine_similarity_self():
    X = np.random.rand(DIM1, DIM2)
    X_list = X.tolist()
    assert np.allclose(cosine_similarity(X, X), cosine_similarity_pure(X_list, X_list))


def test_cosine_distances():
    X = np.random.rand(DIM1, DIM2)
    Y = np.random.rand(DIM1, DIM2)
    X_list = X.tolist()
    Y_list = Y.tolist()
    assert np.allclose(cosine_distances(X, Y), cosine_distances_pure(X_list, Y_list))


def test_cosine_distances_self():
    X = np.random.rand(DIM1, DIM2)
    X_list = X.tolist()
    assert np.allclose(cosine_distances(X, X), cosine_distances_pure(X_list, X_list))


def test_linear_kernel():
    X = np.random.rand(DIM1, DIM2)
    Y = np.random.rand(DIM1, DIM2)
    X_list = X.tolist()
    Y_list = Y.tolist()
    assert np.allclose(linear_kernel(X, Y), linear_kernel_pure(X_list, Y_list))


def test_linear_kernel_self():
    X = np.random.rand(DIM1, DIM2)
    X_list = X.tolist()
    assert np.allclose(linear_kernel(X, X), linear_kernel_pure(X_list, X_list))
