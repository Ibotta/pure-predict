"""
The :mod:`pure_sklearn.ensemble` module implements a variety of ensemble models
"""

from ._forest import RandomForestClassifierPure, ExtraTreesClassifierPure
from ._bagging import BaggingClassifierPure
from ._gb import GradientBoostingClassifierPure

__all__ = [
    "RandomForestClassifierPure",
    "BaggingClassifierPure",
    "ExtraTreesClassifierPure",
    "GradientBoostingClassifierPure",
]
