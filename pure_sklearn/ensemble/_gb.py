"""
Gradient boosting
"""

from operator import add

from ._gb_losses import (
    _MultinomialDeviancePure,
    _BinomialDeviancePure,
    _ExponentialLossPure,
)

from ..base import transpose, apply_2d, safe_log, operate_2d
from ..utils import check_version, check_types, check_array, shape
from ..map import convert_estimator


class GradientBoostingClassifierPure:
    """
    Pure python implementation of `GradientBoostingClassifier`.

    Args:
        estimator (sklearn estimator): fitted `GradientBoostingClassifier` object
    """

    def __init__(self, estimator):
        check_version(estimator, "0.21.0")
        self.classes_ = estimator.classes_.tolist()
        self.estimators_ = []
        for est_arr in estimator.estimators_:
            est_arr_ = []
            for est in est_arr:
                est_ = convert_estimator(est)
                est_arr_.append(est_)
            self.estimators_.append(est_arr_)
        if hasattr(estimator, "init_"):
            self.init_ = convert_estimator(estimator.init_)
        self.loss = estimator.loss
        self.learning_rate = estimator.learning_rate
        self.n_features_ = estimator.n_features_
        if self.loss == "deviance":
            self.loss_ = (
                _MultinomialDeviancePure(len(self.classes_))
                if len(self.classes_) > 2
                else _BinomialDeviancePure(len(self.classes_))
            )
        elif self.loss == "exponential":
            self.loss_ = _ExponentialLossPure(len(self.classes_))
        else:
            raise ValueError("Loss: '{}' not supported.".format(self.loss))
        check_types(self)

    def _raw_predict_init(self, X):
        """ Check input and compute raw predictions of the init estimator """
        X = check_array(X)
        if shape(X)[1] != self.n_features_:
            raise ValueError(
                "X.shape[1] should be {0:d}, not {1:d}.".format(
                    self.n_features_, shape(X)[1]
                )
            )
        if self.init_ == "zero":
            raw_predictions = [[0.0] * shape(X)[1]] * shape(X)[0]
        else:
            raw_predictions = self.loss_.get_init_raw_predictions(X, self.init_)
        return raw_predictions

    def _raw_predict(self, X):
        init_preds = self._raw_predict_init(X)
        out = []
        for k in range(len(self.estimators_[0])):
            column = [0] * (shape(X)[0])
            for index in range(len(self.estimators_)):
                preds = self.estimators_[index][k].predict(X)
                column = [
                    column[i] + (preds[i] * self.learning_rate)
                    for i in range(len(preds))
                ]
            out.append(column)
        out = transpose(out)
        return operate_2d(init_preds, out, add)

    def predict_proba(self, X):
        raw_predictions = self._raw_predict(X)
        return self.loss_._raw_prediction_to_proba(raw_predictions)

    def predict_log_proba(self, X):
        return apply_2d(self.predict_proba(X), safe_log)

    def predict(self, X):
        proba = self.predict_proba(X)
        return [self.classes_[a.index(max(a))] for a in proba]
