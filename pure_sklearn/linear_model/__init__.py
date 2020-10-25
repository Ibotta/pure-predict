"""
The :mod:`pure_sklearn.linear_model` module implements a variety of linear models
"""

from ._base import LinearClassifierMixinPure
from ._logistic import LogisticRegressionPure
from ._ridge import RidgeClassifierPure
from ._stochastic_gradient import SGDClassifierPure
from ._perceptron import PerceptronPure
from ._passive_aggressive import PassiveAggressiveClassifierPure

__all__ = [
    "LogisticRegressionPure",
    "RidgeClassifierPure",
    "SGDClassifierPure",
    "PerceptronPure",
    "PassiveAggressiveClassifierPure",
    "LinearClassifierMixinPure",
]
