"""
Base functions for matrix calculations
"""

from math import exp, log
from operator import mul

from .utils import shape, check_array, sparse_list, todense, issparse


def dot(A, B):
    """
    Dot product between two arrays.
    A -> n_dim = 1
    B -> n_dim = 2
    """
    arr = []
    for i in range(len(B)):
        if isinstance(A, dict):
            val = sum([v * B[i][k] for k, v in A.items()])
        else:
            val = sum(map(mul, A, B[i]))
        arr.append(val)
    return arr


def dot_2d(A, B):
    """
    Dot product between two arrays.
    A -> n_dim = 2
    B -> n_dim = 2
    """
    return [dot(a, B) for a in A]


def matmult_same_dim(A, B):
    """ Multiply two matrices of the same dimension """
    shape_A = shape(A)
    shape_B = shape(B)
    issparse_A = issparse(A)
    issparse_B = issparse(B)
    if shape_A != shape(B):
        raise ValueError("Shape A must equal shape B.")
    if not (issparse_A == issparse_B):
        raise ValueError("Both A and B must be sparse or dense.")

    X = []
    if not issparse_A:
        for i in range(shape_A[0]):
            X.append([(A[i][j] * B[i][j]) for j in range(shape_A[1])])
    else:
        for i in range(shape_A[0]):
            nested_res = [
                [(k_b, v_a * v_b) for k_b, v_b in B[i].items() if k_b == k_a]
                for k_a, v_a in A[i].items()
            ]
            X.append(dict([item for sublist in nested_res for item in sublist]))
        X = sparse_list(X, size=A.size, dtype=A.dtype)
    return X


def transpose(A):
    """ Transpose 2-D list """
    if issparse(A):
        raise ValueError("Sparse input not supported.")
    return list(map(list, [*zip(*A)]))


def expit(x):
    """ Expit function for scaler input """
    return 1.0 / (1.0 + safe_exp(-x))


def sfmax(arr):
    """ Softmax function for 1-D list or a single sparse_list element """
    if isinstance(arr, dict):
        expons = {k: safe_exp(v) for k, v in arr.items()}
        denom = sum(expons.values())
        out = {k: (v / float(denom)) for k, v in expons.items()}
    else:
        expons = list(map(safe_exp, arr))
        out = list(map(lambda x: x / float(sum(expons)), expons))
    return out


def safe_log(x):
    """ Equivalent to numpy log with scalar input """
    if x == 0:
        return -float("Inf")
    elif x < 0:
        return float("Nan")
    else:
        return log(x)


def safe_exp(x):
    """ Equivalent to numpy exp with scalar input """
    try:
        return exp(x)
    except OverflowError:
        return float("Inf")


def operate_2d(A, B, func):
    """ Apply elementwise function to 2-D lists """
    if issparse(A) or issparse(B):
        raise ValueError("Sparse input not supported.")
    if shape(A) != shape(B):
        raise ValueError("'A' and 'B' must have the same shape")
    return [list(map(func, A[index], B[index])) for index in range(len(A))]


def apply_2d(A, func):
    """ Apply function to every element of 2-D list """
    if issparse(A):
        raise ValueError("Sparse input not supported.")
    return [list(map(func, a)) for a in A]


def apply_2d_sparse(A, func):
    """ Apply function to every non-zero element of sparse_list """
    if not issparse(A):
        raise ValueError("Dense input not supported.")
    A_ = [{k: func(v) for k, v in a.items()} for a in A]
    return sparse_list(A_, size=A.size, dtype=A.dtype)


def apply_axis_2d(A, func, axis=1):
    """
    Apply function along axis of 2-D list or non-zero
    elements of sparse_list.
    """
    if issparse(A) and (axis == 0):
        raise ValueError("Sparse input not supported when axis=0.")
    if axis == 1:
        if issparse(A):
            return [func(a.values()) for a in A]
        else:
            return [func(a) for a in A]
    elif axis == 0:
        return [func(a) for a in transpose(A)]
    else:
        raise ValueError("Input 'axis' must be 0 or 1")


def ravel(A):
    """ Equivalent of numpy ravel on 2-D list """
    if issparse(A):
        raise ValueError("Sparse input not supported.")
    return list(transpose(A)[0])


def slice_column(A, idx):
    """ Slice columns from 2-D list A. Handles sparse data """
    if isinstance(idx, int):
        if issparse(A):
            return [a.get(idx, A.dtype(0)) for a in A]
        else:
            return [a[idx] for a in A]
    if isinstance(idx, (list, tuple)):
        if issparse(A):
            A_ = [{k: v for k, v in a.items() if k in idx} for a in A]
            return sparse_list(A_, size=A.size, dtype=A.dtype)
        else:
            return [[a[i] for i in idx] for a in A]


def accumu(lis):
    """ Cumulative sum of list """
    total = 0
    for x in lis:
        total += x
        yield total
