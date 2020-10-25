"""
Pairwise distance and similarity metrics
"""

from math import sqrt

from ..base import dot_2d, apply_2d, transpose
from ..utils import shape, ndim, issparse
from ..preprocessing import normalize_pure

__all__ = ["cosine_similarity_pure", "cosine_distances_pure", "linear_kernel_pure"]


def _clip(a, a_min, a_max):
    if a < a_min:
        return a_min
    elif a > a_max:
        return a_max
    else:
        return a


def _set_diag(S, val=0.0):
    for i in range(len(S)):
        for j in range(len(S[i])):
            if i == j:
                S[i][j] = val
    return S


def _check_pairwise_arrays(X, Y):
    if Y is None:
        Y = X
    if (ndim(X) != 2) or (ndim(Y) != 2):
        raise ValueError("Input arrays must be 2-D.")
    if shape(X)[1] != shape(Y)[1]:
        raise ValueError("Input arrays must have same 2nd dimension.")
    if issparse(Y):
        raise ValueError("Input array 'Y' cannot be sparse.")
    return X, Y


def cosine_similarity_pure(X, Y=None):
    """ Compute cosine similarity between samples in X and Y """
    X, Y = _check_pairwise_arrays(X, Y)
    X_normalized = normalize_pure(X, copy=True)
    if X is Y:
        Y_normalized = X_normalized
    else:
        Y_normalized = normalize_pure(Y, copy=True)
    K = dot_2d(X_normalized, Y_normalized)
    return K


def cosine_distances_pure(X, Y=None):
    """ Compute cosine distance between samples in X and Y """
    S = cosine_similarity_pure(X, Y)
    func = lambda x: _clip(-x + 1, 0, 2)
    S = apply_2d(S, func)
    if X is Y or Y is None:
        S = _set_diag(S)
    return S


def linear_kernel_pure(X, Y=None):
    """ Compute the linear kernel between X and Y """
    X, Y = _check_pairwise_arrays(X, Y)
    return dot_2d(X, Y)
