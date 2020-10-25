"""
Feature encoders
"""

from ..utils import (
    check_types,
    check_array,
    shape,
    sparse_list,
    convert_type,
    check_version,
)
from ..base import accumu, apply_2d, ravel
from ._label import _encode, _encode_check_unknown


class _BaseEncoderPure:
    """
    Base class for encoders that includes the code to categorize and
    transform the input features.
    """

    def __init__(self, estimator):
        check_version(estimator)
        self.dtype = convert_type(estimator.dtype)
        self.categories_ = [a.tolist() for a in estimator.categories_]
        if hasattr(estimator, "sparse"):
            self.sparse = estimator.sparse
        if hasattr(estimator, "drop") and (estimator.drop is not None):
            raise ValueError("Encoder does not handle 'drop' functionality")
        if hasattr(estimator, "handle_unknown"):
            self.handle_unknown = estimator.handle_unknown
        check_types(self)

    def _check_X(self, X):
        """ Perform custom check_array """
        X = check_array(X)
        n_samples, n_features = shape(X)
        X_columns = []
        for i in range(n_features):
            Xi = self._get_feature(X, feature_idx=i)
            X_columns.append(Xi)
        return X_columns, n_samples, n_features

    def _get_feature(self, X, feature_idx):
        return [x[feature_idx] for x in X]

    def _transform(self, X, handle_unknown="error"):
        X_list, n_samples, n_features = self._check_X(X)
        X_int = [[0] * n_features] * n_samples
        X_mask = [[True] * n_features] * n_samples

        if n_features != len(self.categories_):
            raise ValueError(
                "The number of features in X is different to the number of "
                "features of the fitted data. The fitted data had {} features "
                "and the X has {} features.".format(
                    len(
                        self.categories_,
                    ),
                    n_features,
                )
            )

        for i in range(n_features):
            Xi = X_list[i]
            diff, valid_mask = _encode_check_unknown(
                Xi, self.categories_[i], return_mask=True
            )

            if not (sum(valid_mask) == len(valid_mask)):
                if handle_unknown == "error":
                    msg = (
                        "Found unknown categories {0} in column {1}"
                        " during transform".format(diff, i)
                    )
                    raise ValueError(msg)
                else:
                    X_mask = [
                        [
                            valid_mask[j] if idx == i else X_mask[j][idx]
                            for idx in range(n_features)
                        ]
                        for j in range(n_samples)
                    ]
                    Xi = [
                        Xi[idx] if valid_mask[idx] else self.categories_[i][0]
                        for idx in range(len(Xi))
                    ]

            _, encoded = _encode(
                Xi, self.categories_[i], encode=True, check_unknown=False
            )
            X_int = [
                [encoded[j] if idx == i else X_int[j][idx] for idx in range(n_features)]
                for j in range(n_samples)
            ]
        return X_int, X_mask


class OrdinalEncoderPure(_BaseEncoderPure):
    """
    Pure python implementation of `OrdinalEncoder`.

    Args:
        estimator (sklearn estimator): fitted `OrdinalEncoder` object
    """

    def transform(self, X):
        """ Transform X to ordinal codes """
        X_int, _ = self._transform(X)
        return apply_2d(X_int, self.dtype)


class OneHotEncoderPure(_BaseEncoderPure):
    """
    Pure python implementation of `OneHotEncoder`.

    Args:
        estimator (sklearn estimator): fitted `OneHotEncoder` object
    """

    def transform(self, X):
        """ Transform X using one-hot encoding """
        X_int, X_mask = self._transform(X, handle_unknown=self.handle_unknown)

        n_samples, n_features = shape(X_int)
        n_values = [0] + [len(cats) for cats in self.categories_]
        feature_indices = list(accumu(n_values))
        data = [
            dict(
                [
                    (n_values[i] + X_int[j][i], self.dtype(1))
                    for i in range(n_features)
                    if X_mask[j][i]
                ]
            )
            for j in range(n_samples)
        ]
        out = sparse_list(data, size=feature_indices[-1], dtype=self.dtype)
        if not self.sparse:
            return out.todense()
        else:
            return out
