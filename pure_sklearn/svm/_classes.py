"""
Support vector machines
"""

from ..utils import check_types, check_version
from ..linear_model import LinearClassifierMixinPure


class LinearSVCPure(LinearClassifierMixinPure):
    """
    Pure python implementation of `LinearSVC`.

    Args:
        estimator (sklearn estimator): fitted `LinearSVC` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        super().__init__(estimator=estimator)
        check_types(self)
