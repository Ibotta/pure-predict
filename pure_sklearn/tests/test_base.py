import warnings
import numpy as np

from operator import add

from pure_sklearn.utils import shape, tosparse
from pure_sklearn.base import (
    transpose,
    dot,
    safe_log,
    safe_exp,
    dot_2d,
    matmult_same_dim,
    sfmax,
    operate_2d,
    apply_2d,
    apply_2d_sparse,
    apply_axis_2d,
    slice_column,
)
from pure_sklearn.utils import tosparse

A = [1, 2]
B = [[1, 2], [3, 4]]
LOG_LIST = [-float("Inf"), -1e30, -10, 0, 10, 1e30, 1e500, float("Inf"), float("Nan")]


def test_import():
    from pure_sklearn import base

    assert True


def test_transpose():
    assert transpose(B) == np.array(B).T.tolist()
    assert shape(transpose(B)) == shape(B)[::-1]


def test_dot():
    assert dot(A, B) == [5, 11]


def test_dot_sparse():
    A_sparse = tosparse([A])[0]
    assert dot(A_sparse, B) == [5, 11]


def test_dot_2d():
    assert dot_2d([A], B) == [[5, 11]]


def test_dot_2d_sparse():
    A_sparse = tosparse([A])
    assert dot_2d(A_sparse, B) == [[5, 11]]


def test_safe_exp():
    for n in LOG_LIST:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert np.allclose(np.exp(n), safe_exp(n), equal_nan=True)


def test_safe_log():
    for n in LOG_LIST:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert np.allclose(np.exp(n), safe_exp(n), equal_nan=True)


def test_matmult():
    X = matmult_same_dim(B, B)
    assert np.allclose(X, [[1, 4], [9, 16]])


def test_matmult_sparse():
    sparse_B = tosparse(B)
    X = matmult_same_dim(sparse_B, sparse_B)
    assert np.allclose(X.todense(), [[1, 4], [9, 16]])


def test_sfmax():
    np.allclose(sfmax(A), [0.26894142136999, 0.73105857863000])


def test_sfmax_sparse():
    A_sparse = tosparse([A])[0]
    lst = sfmax(A)
    assert sfmax(A_sparse) == {0: lst[0], 1: lst[1]}


def test_operate_2d():
    assert operate_2d(B, B, add) == [[2, 4], [6, 8]]


def test_apply_2d():
    assert apply_2d(B, lambda x: 2 * x) == [[2, 4], [6, 8]]


def test_apply_2d_sparse():
    B_sparse = tosparse(B)
    B_applied = apply_2d_sparse(B_sparse, lambda x: 2 * x).todense()
    assert B_applied == [[2, 4], [6, 8]]


def test_apply_axis_2d():
    assert apply_axis_2d(B, sum) == [3, 7]


def test_apply_axis_2d_sparse():
    B_sparse = tosparse(B)
    assert apply_axis_2d(B_sparse, sum) == [3, 7]


def test_slice():
    assert slice_column(B, 0) == [1, 3]


def test_slice_sparse():
    assert slice_column(tosparse(B), 0) == [1, 3]


def test_slice_array():
    assert slice_column(B, [0, 1]) == B


def test_slice_array_sparse():
    assert slice_column(tosparse(B), [0, 1]).todense() == B
