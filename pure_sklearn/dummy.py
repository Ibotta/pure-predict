"""
Dummy classifier
"""

from .base import apply_2d, safe_log
from .utils import check_types, check_version

__all__ = ["DummyClassifierPure"]


class DummyClassifierPure:
    """
    Pure python implementation of `DummyClassifier`. Only supports
    'prior' strategy.

    Args:
        estimator (sklearn estimator): fitted `DummyClassifier` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        if hasattr(estimator, "_strategy"):
            self._strategy = estimator._strategy
        else:
            self._strategy = estimator.strategy
        if self._strategy != "prior":
            raise ValueError("Strategy '{}' not supported".format(self._strategy))
        self.class_prior_ = estimator.class_prior_.tolist()
        self.n_classes_ = estimator.n_classes_
        self.n_outputs_ = estimator.n_outputs_
        check_types(self)

    def predict(self, X):
        return [self.class_prior_.index(max(self.class_prior_))] * len(X)

    def predict_proba(self, X):
        return [self.class_prior_] * len(X)

    def predict_log_proba(self, X):
        return apply_2d(self.predict_proba(X), safe_log)
