"""
Normalization and scaling
"""

from math import sqrt
from copy import copy as cp

from ..utils import sparse_list, issparse, check_array, check_types, check_version
from ..base import transpose, apply_2d, apply_axis_2d, matmult_same_dim


def _handle_zeros_in_scale(scale, copy=True):
    """ Makes sure that whenever scale is zero, we handle it correctly """
    if isinstance(scale, (int, float)):
        if scale == 0.0:
            scale = 1.0
        return scale
    elif isinstance(scale, list):
        if copy:
            scale = cp(scale)
        return [(1.0 if scale[i] == 0.0 else scale[i]) for i in range(len(scale))]


def _row_norms(X):
    """ Row-wise (squared) Euclidean norm of X """
    X_X = matmult_same_dim(X, X)
    if issparse(X):
        norms = [sum(x.values()) for x in X_X]
    else:
        norms = apply_axis_2d(X_X, sum, axis=1)
    return list(map(sqrt, norms))


def normalize_pure(X, norm="l2", axis=1, copy=True, return_norm=False):
    """ Scale input vectors individually to unit norm """
    # check input compatibility
    if (axis == 0) and issparse(X):
        raise ValueError("Axis 0 is not supported for sparse data")
    if norm not in ("l1", "l2", "max"):
        raise ValueError("'%s' is not a supported norm" % norm)
    if axis not in [0, 1]:
        raise ValueError("'%d' is not a supported axis" % axis)
    X = check_array(X, handle_sparse="allow")

    if axis == 0:
        X = transpose(X)

    if issparse(X):
        if return_norm and norm in ("l1", "l2"):
            raise NotImplementedError(
                "return_norm=True is not implemented "
                "for sparse matrices with norm 'l1' "
                "or norm 'l2'"
            )
        if norm == "l1":
            norms = [sum(map(abs, x.values())) for x in X]
        elif norm == "l2":
            norms = _row_norms(X)
        elif norm == "max":
            norms = [max(list(x.values()) + [0]) for x in X]
        norms = _handle_zeros_in_scale(norms, copy=False)
        X_sparse = [
            {k: (v / float(norms[index])) for k, v in X[index].items()}
            for index in range(len(X))
        ]
        X = sparse_list(X_sparse, X.size, X.dtype)
    else:
        if norm == "l1":
            norms = apply_axis_2d(apply_2d(X, abs), sum, axis=1)
        elif norm == "l2":
            norms = _row_norms(X)
        elif norm == "max":
            norms = apply_axis_2d(X, max, axis=1)
        norms = _handle_zeros_in_scale(norms, copy=False)
        X = [
            list(map(lambda a: a / float(norms[index]), X[index]))
            for index in range(len(X))
        ]

    if axis == 0:
        X = transpose(X)

    if return_norm:
        return X, norms
    else:
        return X


class NormalizerPure:
    """
    Pure python implementation of `Normalizer`.

    Args:
        estimator (sklearn estimator): fitted `Normalizer` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        self.norm = estimator.norm
        self.copy = estimator.copy
        check_types(self)

    def transform(self, X, copy=None):
        """Scale each non zero row of X to unit norm."""
        copy = copy if copy is not None else self.copy
        X = check_array(X, handle_sparse="allow")
        return normalize_pure(X, norm=self.norm, axis=1, copy=copy)


class StandardScalerPure:
    """
    Pure python implementation of `StandardScaler`.

    Args:
        estimator (sklearn estimator): fitted `StandardScaler` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        self.with_mean = estimator.with_mean
        self.with_std = estimator.with_std
        if estimator.scale_ is None:
            self.scale_ = None
        else:
            self.scale_ = estimator.scale_.tolist()
        if estimator.mean_ is None:
            self.mean_ = None
        else:
            self.mean_ = estimator.mean_.tolist()
        check_types(self)

    def transform(self, X, copy=None):
        """ Perform standardization by centering and scaling """
        X = check_array(X, handle_sparse="allow")

        if issparse(X):
            if self.with_mean:
                raise ValueError(
                    "Cannot center sparse matrices: pass `with_mean=False` "
                    "instead. See docstring for motivation and alternatives."
                )
            if self.scale_ is not None:
                X_ = [{k: (v / self.scale_[k]) for k, v in x.items()} for x in X]
                X = sparse_list(X_, size=X.size, dtype=X.dtype)
        else:
            if self.with_mean:
                X = [[x[i] - self.mean_[i] for i in range(len(self.mean_))] for x in X]
            if self.with_std:
                X = [
                    [x[i] / self.scale_[i] for i in range(len(self.scale_))] for x in X
                ]
        return X


class MinMaxScalerPure:
    """
    Pure python implementation of `MinMaxScaler`.

    Args:
        estimator (sklearn estimator): fitted `MinMaxScaler` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        self.feature_range = estimator.feature_range
        if estimator.scale_ is None:
            self.scale_ = None
        else:
            self.scale_ = estimator.scale_.tolist()
        if estimator.min_ is None:
            self.min_ = None
        else:
            self.min_ = estimator.min_.tolist()
        check_types(self)

    def transform(self, X):
        """ Scale features of X according to feature_range """
        if issparse(X):
            raise TypeError(
                "MinMaxScalerPure does not support sparse input. "
                "Consider using MaxAbsScalerPure instead."
            )
        X = check_array(X)
        return [
            [(x[i] * self.scale_[i]) + self.min_[i] for i in range(len(self.scale_))]
            for x in X
        ]


class MaxAbsScalerPure:
    """
    Pure python implementation of `MaxAbsScaler`.

    Args:
        estimator (sklearn estimator): fitted `MaxAbsScaler` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        self.copy = estimator.copy
        if estimator.scale_ is None:
            self.scale_ = None
        else:
            self.scale_ = estimator.scale_.tolist()
        if estimator.max_abs_ is None:
            self.max_abs_ = None
        else:
            self.max_abs_ = estimator.max_abs_.tolist()
        check_types(self)

    def transform(self, X):
        """ Scale the data """
        X = check_array(X, handle_sparse="allow")
        if issparse(X):
            X_ = [{k: (v / self.scale_[k]) for k, v in x.items()} for x in X]
            X = sparse_list(X_, size=X.size, dtype=X.dtype)
        else:
            X = [[x[i] / self.scale_[i] for i in range(len(self.scale_))] for x in X]
        return X
