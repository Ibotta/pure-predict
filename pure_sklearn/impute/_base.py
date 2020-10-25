"""
Imputer transformers
"""

from math import isnan

from ..utils import shape, check_array, check_types, check_version
from ..base import apply_2d, apply_axis_2d


def _to_impute(val, missing_values):
    if isnan(missing_values):
        return isnan(val)
    else:
        return val == missing_values


class MissingIndicatorPure:
    """
    Pure python implementation of `MissingIndicator`.

    Args:
        estimator (sklearn estimator): fitted `MissingIndicator` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        self.features = estimator.features
        self.features_ = estimator.features_.tolist()
        self._n_features = estimator._n_features
        self.missing_values = (
            float(estimator.missing_values)
            if isinstance(estimator.missing_values, float)
            else estimator.missing_values
        )
        self.error_on_new = estimator.error_on_new
        check_types(self)

    def transform(self, X):
        X = check_array(X)
        if shape(X)[1] != self._n_features:
            raise ValueError(
                "X has a different number of features than during fitting."
            )

        imputer_mask, features = self._get_missing_features_info(X)

        if self.features == "missing-only":
            features_diff_fit_trans = set(features) - set(self.features_)
            if self.error_on_new and len(features_diff_fit_trans) > 0:
                raise ValueError(
                    "The features {} have missing values "
                    "in transform but have no missing values "
                    "in fit.".format(features_diff_fit_trans)
                )

            if len(self.features_) < self._n_features:
                imputer_mask = [
                    [float(a[i]) for i in range(len(a)) if i in self.features_]
                    for a in imputer_mask
                ]
        return imputer_mask

    def _get_missing_features_info(self, X):
        func = lambda x: _to_impute(x, self.missing_values)
        imputer_mask = apply_2d(X, func)

        if self.features == "missing-only":
            n_missing = apply_axis_2d(imputer_mask, sum, axis=0)
        if self.features == "all":
            features_indices = range(shape(X)[1])
        else:
            features_indices = [a for a in n_missing if a != 0]
        return imputer_mask, features_indices


class SimpleImputerPure:
    """
    Pure python implementation of `SimpleImputer`.

    Args:
        estimator (sklearn estimator): fitted `SimpleImputer` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        self.statistics_ = estimator.statistics_.tolist()
        self.strategy = estimator.strategy
        if hasattr(estimator, "add_indicator"):
            self.add_indicator = estimator.add_indicator
        else:
            self.add_indicator = False
        self.missing_values = (
            float(estimator.missing_values)
            if isinstance(estimator.missing_values, float)
            else estimator.missing_values
        )
        if hasattr(estimator, "indicator_") and (estimator.indicator_ is not None):
            self.indicator_ = MissingIndicatorPure(estimator.indicator_)
            self.indicator_.error_on_new = False
        check_types(self)

    def _concatenate_indicator(self, X_imputed, X_indicator):
        """ Concatenate indicator mask with the imputed data """
        if not self.add_indicator:
            return X_imputed

        if X_indicator is None:
            raise ValueError(
                "Data from the missing indicator are not provided. Call "
                "_fit_indicator and _transform_indicator in the imputer "
                "implementation."
            )
        return [
            X_imputed[index] + X_indicator[index] for index in range(len(X_imputed))
        ]

    def _transform_indicator(self, X):
        """
        Compute the indicator mask.
        Note that X must be the original data as passed to the imputer before
        any imputation, since imputation may be done inplace in some cases.
        """
        if self.add_indicator:
            if not hasattr(self, "indicator_"):
                raise ValueError(
                    "Make sure to call _fit_indicator before _transform_indicator"
                )
            return self.indicator_.transform(X)

    def transform(self, X):
        """ Transform inpute X by imputing values """
        X = check_array(X)
        X_indicator = self._transform_indicator(X)

        if shape(X)[1] != shape(self.statistics_)[0]:
            raise ValueError(
                "X has %d features per sample, expected %d"
                % (shape(X)[1], shape(self.statistics_)[0])
            )

        # delete the invalid columns if strategy is not constant
        if self.strategy == "constant":
            valid_statistics = self.statistics_
        else:
            to_remove = [
                index
                for index in range(len(self.statistics_))
                if isnan(self.statistics_[index])
            ]
            if len(to_remove) > 0:
                X = [[a[i] for i in range(len(a)) if i not in to_remove] for a in X]
                valid_statistics = [
                    self.statistics_[i]
                    for i in range(len(self.statistics_))
                    if i not in to_remove
                ]
            else:
                valid_statistics = self.statistics_

        func = (
            lambda a, i: a[i]
            if not _to_impute(a[i], self.missing_values)
            else valid_statistics[i]
        )
        X_imputed = [[func(a, i) for i in range(len(a))] for a in X]
        return self._concatenate_indicator(X_imputed, X_indicator)
