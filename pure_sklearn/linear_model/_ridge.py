"""
Ridge Regression
"""

from ..utils import check_types, check_version
from ._base import LinearClassifierMixinPure


class RidgeClassifierPure(LinearClassifierMixinPure):
    """
    Pure python implementation of `RidgeClassifier`.

    Args:
        estimator (sklearn estimator): fitted `RidgeClassifier` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        super().__init__(estimator=estimator)
        check_types(self)
