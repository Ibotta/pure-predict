"""
Estimator classes for xgboost
"""

from math import exp

from ..base import sfmax, expit
from ..utils import check_types, check_array
from ..tree import DecisionTreeRegressorPure

MIN_VERSION = "0.82"
SUPPORTED_OBJ = ["binary:logistic", "multi:softprob"]
SUPPORTED_BOOSTER = ["gbtree"]


class XGBClassifierPure:
    """
    Pure python implementation of `XGBClassifier`. Only supports 'gbtree'
    booster and 'binary:logistic' or 'multi:softprob' objectives.

    Args:
        estimator (xgboost estimator): fitted `XGBClassifier` object
    """

    def __init__(self, estimator):
        if (not isinstance(estimator.objective, str)) or (
            estimator.objective not in SUPPORTED_OBJ
        ):
            raise ValueError(
                "Objective function not supported; only {} are supported".format(
                    SUPPORTED_OBJ
                )
            )
        else:
            self.objective = estimator.objective
        if estimator.booster not in SUPPORTED_BOOSTER:
            raise ValueError("Booster: '{}' not supported".format(estimator.booster))
        else:
            self.booster = estimator.booster
        self.classes_ = estimator.classes_.tolist()
        self.n_classes_ = estimator.n_classes_
        self.n_estimators = estimator.n_estimators
        self.estimators_ = self._build_estimators(estimator)
        check_types(self)

    def _build_estimators(self, estimator):
        """ Convert booster to list of pure decision tree regressors """
        if not hasattr(estimator.get_booster(), "trees_to_dataframe"):
            raise Exception(
                "This xgboost estimator was likely fitted with version < {} "
                "which is not supported".format(MIN_VERSION)
            )
        tree_df = estimator.get_booster().trees_to_dataframe()
        estimators_ = []
        idx = 0
        for est_id in range(self.n_estimators):
            if self.n_classes_ == 2:
                tree = tree_df[tree_df["Tree"] == idx].to_dict(orient="list")
                est_row_ = DecisionTreeRegressorPure(tree)
                idx += 1
            else:
                est_row_ = []
                for cls_id in range(self.n_classes_):
                    tree = tree_df[tree_df["Tree"] == idx].to_dict(orient="list")
                    est_row_.append(DecisionTreeRegressorPure(tree))
                    idx += 1
            estimators_.append(est_row_)
        return estimators_

    def _predict(self, X):
        """ Raw sums of estimator predictions for each class for multi-class """
        preds = []
        for cls_index in range(self.n_classes_):
            cls_sum = [0] * len(X)
            for est_index in range(self.n_estimators):
                est_preds = self.estimators_[est_index][cls_index].predict(X)
                cls_sum = list(map(lambda x, y: x + y, cls_sum, est_preds))
            preds.append(cls_sum)
        return preds

    def _predict_binary(self, X):
        """ Raw sums of estimator predictions for each class for binary """
        preds = [0] * len(X)
        for estimator in self.estimators_:
            preds = list(map(lambda x, y: x + y, preds, estimator.predict(X)))
        return preds

    def predict(self, X):
        proba = self.predict_proba(X)
        return [self.classes_[a.index(max(a))] for a in proba]

    def predict_proba(self, X):
        X = check_array(X)
        if self.objective == "multi:softprob":
            preds = self._predict(X)
            out = []
            for i in range(len(X)):
                out.append(sfmax([preds[j][i] for j in range(self.n_classes_)]))
        elif self.objective == "binary:logistic":
            preds = self._predict_binary(X)
            out = list(map(expit, preds))
            out = list(map(lambda x: [1 - x, x], out))
        else:
            raise ValueError(
                "Objective function not supported; only {} are supported".format(
                    SUPPORTED_OBJ
                )
            )
        return out
