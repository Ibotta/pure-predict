"""
Gradient boosting loss classes
"""

from ..base import transpose, apply_axis_2d, apply_2d, safe_exp, safe_log, ravel, expit
from ..utils import check_types, shape

EPS = 1.1920929e-07


def _clip(a, a_min, a_max):
    if a < a_min:
        return a_min
    elif a > a_max:
        return a_max
    else:
        return a


class _MultinomialDeviancePure:
    """ Multinomial deviance loss function for multi-class classification """

    is_multi_class = True

    def __init__(self, n_classes):
        if n_classes < 3:
            raise ValueError(
                "{0:s} requires more than 2 classes.".format(self.__class__.__name__)
            )
        self.n_classes_ = n_classes
        check_types(self)

    def _raw_prediction_to_proba(self, raw_predictions):
        logsumexp = list(
            map(safe_log, apply_axis_2d(apply_2d(raw_predictions, safe_exp), sum))
        )
        return [
            [
                safe_exp(raw_predictions[index][i] - logsumexp[index])
                for i in range(self.n_classes_)
            ]
            for index in range(len(raw_predictions))
        ]

    def _raw_prediction_to_decision(self, raw_predictions):
        proba = self._raw_prediction_to_proba(raw_predictions)
        return [a.index(max(a)) for a in proba]

    def get_init_raw_predictions(self, X, estimator):
        probas = estimator.predict_proba(X)
        func = lambda x: safe_log(_clip(x, EPS, 1 - EPS))
        return apply_2d(probas, func)


class _BinomialDeviancePure:
    """ Binomial deviance loss function for binary classification """

    def __init__(self, n_classes):
        if n_classes != 2:
            raise ValueError(
                "{0:s} requires 2 classes; got {1:d} class(es)".format(
                    self.__class__.__name__, n_classes
                )
            )
        check_types(self)

    def _raw_prediction_to_proba(self, raw_predictions):
        proba = (
            ravel(raw_predictions)
            if shape(raw_predictions)[1] == 1
            else raw_predictions
        )
        proba_1 = list(map(expit, proba))
        proba = [[(1 - x) for x in proba_1], proba_1]
        return transpose(proba)

    def _raw_prediction_to_decision(self, raw_predictions):
        proba = self._raw_prediction_to_proba(raw_predictions)
        return [a.index(max(a)) for a in proba]

    def get_init_raw_predictions(self, X, estimator):
        probas = estimator.predict_proba(X)
        func = lambda x: _clip(x, EPS, 1 - EPS)
        proba_pos_class = [func(a[1]) for a in probas]
        log_func = lambda x: safe_log(x / (1 - x))
        return [[log_func(a)] for a in proba_pos_class]


class _ExponentialLossPure:
    """ Exponential loss function for binary classification """

    def __init__(self, n_classes):
        if n_classes != 2:
            raise ValueError(
                "{0:s} requires 2 classes; got {1:d} class(es)".format(
                    self.__class__.__name__, n_classes
                )
            )
        check_types(self)

    def _raw_prediction_to_proba(self, raw_predictions):
        proba = (
            ravel(raw_predictions)
            if shape(raw_predictions)[1] == 1
            else raw_predictions
        )
        func = lambda x: expit(x) * 2.0
        proba_1 = list(map(func, proba))
        proba = [[(1 - x) for x in proba_1], proba_1]
        return transpose(proba)

    def _raw_prediction_to_decision(self, raw_predictions):
        raw_predictions = (
            ravel(raw_predictions)
            if shape(raw_predictions)[1] == 1
            else raw_predictions
        )
        return [int(a >= 0) for a in raw_predictions]

    def get_init_raw_predictions(self, X, estimator):
        probas = estimator.predict_proba(X)
        func = lambda x: _clip(x, EPS, 1 - EPS)
        proba_pos_class = [func(a[1]) for a in probas]
        log_func = lambda x: 0.5 * safe_log(x / (1 - x))
        return [[log_func(a)] for a in proba_pos_class]
