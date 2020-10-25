"""
Logistic Regression 
"""

from ._base import LinearClassifierMixinPure
from ..base import sfmax, safe_log
from ..utils import ndim, check_types, check_version


class LogisticRegressionPure(LinearClassifierMixinPure):
    """
    Pure python implementation of `LogisticRegression`.

    Args:
        estimator (sklearn estimator): fitted `LogisticRegression` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        super().__init__(estimator=estimator)
        check_types(self)

    def predict_proba(self, X):
        ovr = self.multi_class in ["ovr", "warn"] or (
            self.multi_class == "auto"
            and (len(self.classes_) <= 2 or self.solver == "liblinear")
        )
        if ovr:
            return super()._predict_proba_lr(X)
        else:
            decision = self.decision_function(X)
            if ndim(decision) == 1:
                decision_2d = [[-a, a] for a in decision]
            else:
                decision_2d = decision
            return list(map(sfmax, decision_2d))

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.
        The returned estimates for all classes are ordered by the
        label of classes.
        """
        return [list(map(safe_log, a)) for a in self.predict_proba(X)]
