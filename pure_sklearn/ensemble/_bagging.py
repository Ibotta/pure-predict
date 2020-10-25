"""
Bagging classifier 
"""

from operator import add

from ..map import MAPPING, convert_estimator
from ..base import safe_log, operate_2d, apply_2d
from ..utils import shape, check_types, check_version


def _feat_slice(X, feats):
    if feats is None:
        return X
    else:
        return [[a[i] for i in feats] for a in X]


class _BaseBaggingPure:
    """ Base bagging classifier """

    def __init__(self, estimator):
        check_version(estimator)
        if hasattr(estimator, "estimators_features_"):
            self.estimators_features_ = [
                a.tolist() for a in estimator.estimators_features_
            ]
        self.classes_ = estimator.classes_.tolist()
        self.estimators_ = []
        for est in estimator.estimators_:
            est_ = convert_estimator(est)
            self.estimators_.append(est_)
        check_types(self)

    def predict_proba(self, X):
        proba = self.estimators_[0].predict_proba(_feat_slice(X, self._get_feats(0)))
        for idx in range(len(self.estimators_))[1:]:
            proba_idx = self.estimators_[idx].predict_proba(
                _feat_slice(X, self._get_feats(idx))
            )
            proba = operate_2d(proba, proba_idx, add)
        func = lambda x: x / float(len(self.estimators_))
        return apply_2d(proba, func)

    def predict_log_proba(self, X):
        return apply_2d(self.predict_proba(X), safe_log)

    def predict(self, X):
        proba = self.predict_proba(X)
        argmax = [a.index(max(a)) for a in proba]
        return [self.classes_[x] for x in argmax]

    def _get_feats(self, idx):
        if hasattr(self, "estimators_features_"):
            return self.estimators_features_[idx]


class BaggingClassifierPure(_BaseBaggingPure):
    """
    Pure python implementation of `BaggingClassifier`.

    Args:
        estimator (sklearn estimator): fitted `BaggingClassifier` object
    """

    pass
