"""
Forest classifiers
"""

from ._bagging import _BaseBaggingPure


class RandomForestClassifierPure(_BaseBaggingPure):
    """
    Pure python implementation of `RandomForestClassifier`.

    Args:
        estimator (sklearn estimator): fitted `RandomForestClassifier` object
    """

    pass


class ExtraTreesClassifierPure(_BaseBaggingPure):
    """
    Pure python implementation of `ExtraTreesClassifier`.

    Args:
        estimator (sklearn estimator): fitted `ExtraTreesClassifier` object
    """

    pass
