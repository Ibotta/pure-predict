import numpy as np

from pure_sklearn.utils import shape
from pure_sklearn.utils import issparse, tosparse, todense, check_array, sparse_list

A = [1, 2]
B = [[1, 2], [3, 4]]


def test_import():
    from pure_sklearn import utils

    assert True


def test_to_sparse_dense():
    assert np.allclose(todense(tosparse(B)), B)


def test_issparse():
    assert issparse(tosparse(B))
    assert not issparse(B)


def test_check_dense():
    assert np.allclose(check_array(A), [A])
    assert np.allclose(check_array(B), B)


def test_check_sparse():
    B_sparse = sparse_list(B)
    assert np.allclose(check_array(B_sparse, handle_sparse="allow").todense(), B)
    caught = False
    try:
        check_array(B_sparse, handle_sparse="error")
    except:
        caught = True
    assert caught
