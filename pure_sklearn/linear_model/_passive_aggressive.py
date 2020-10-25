"""
Passive Aggressive 
"""

from ..utils import check_types, check_version
from ._stochastic_gradient import SGDClassifierPure


class PassiveAggressiveClassifierPure(SGDClassifierPure):
    """
    Pure python implementation of `PassiveAggressiveClassifier`.

    Args:
        estimator (sklearn estimator): fitted `PassiveAggressiveClassifier` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        super().__init__(estimator=estimator)
        check_types(self)
