"""
Perceptron
"""

from ..utils import check_types, check_version
from ._stochastic_gradient import SGDClassifierPure


class PerceptronPure(SGDClassifierPure):
    """
    Pure python implementation of `Perceptron`.

    Args:
        estimator (sklearn estimator): fitted `Perceptron` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        super().__init__(estimator=estimator)
        check_types(self)
