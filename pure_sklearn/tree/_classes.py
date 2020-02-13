"""
Classes for tree based models
"""

import warnings

from math import isnan

from ..base import safe_log
from ..utils import check_array, check_types, check_version

class _DecisionTreeBase():
    """ Decision tree base class """
    def __init__(self, estimator):
        if isinstance(estimator, dict):
            # sourced from xgboost booster object tree dictionary
            self.threshold_ = list(map(lambda x: -2 if isnan(x) else x, estimator["Split"]))
            self.value_ = [[a] for a in estimator["Gain"]]
            self.children_left_ = list(map(lambda x: -1 if not isinstance(x, str) else int(x.split("-")[-1]), estimator["Yes"]))
            self.children_right_ = list(map(lambda x: -1 if not isinstance(x, str) else int(x.split("-")[-1]), estimator["No"]))
            self.feature_ = list(map(lambda x: -2 if x == "Leaf" else int(x.replace("f", "")[-1]), estimator["Feature"]))
        else:
            # sourced from sklearn decision tree
            check_version(estimator)
            self.children_left_ = estimator.tree_.children_left.tolist()
            self.children_right_ = estimator.tree_.children_right.tolist()
            self.feature_ = estimator.tree_.feature.tolist()
            self.threshold_ = estimator.tree_.threshold.tolist()
            self.value_ = [a[0] for a in estimator.tree_.value.tolist()]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if hasattr(estimator, "classes_") and (estimator.classes_ is not None):
                    self.classes_ = estimator.classes_.tolist()
                else:
                    self.classes_ = [0, 1]
        check_types(self)

    def _get_leaf_node(self, x):
        if isinstance(x, dict):
            left_equal = lambda nd: x.get(self.feature_[nd], 0.0)
        else:
            left_equal = lambda nd: x[self.feature_[nd]]
        found_node = False
        node_id = 0
        while (not found_node):
            if (self.children_left_[node_id] == self.children_right_[node_id]):
                found_node = True
            else:
                if left_equal(node_id) <= self.threshold_[node_id]:
                    node_id = self.children_left_[node_id]
                else:
                    node_id = self.children_right_[node_id]
        return node_id

class DecisionTreeClassifierPure(_DecisionTreeBase):
    """
    Pure python implementation of `DecisionTreeClassifier`.

    Args:
        estimator (sklearn estimator): fitted `DecisionTreeClassifier` object
    """
    def _get_pred_from_leaf_node(self, node_id):
        return self.value_[node_id].index(max(self.value_[node_id]))

    def _get_proba_from_leaf_node(self, node_id):
        return [a / sum(self.value_[node_id]) for a in self.value_[node_id]]

    def predict(self, X):
        X = check_array(X, handle_sparse="allow")
        leaves = [self._get_leaf_node(x) for x in X]
        preds = [self._get_pred_from_leaf_node(x) for x in leaves]
        return [self.classes_[x] for x in preds]
    
    def predict_proba(self, X):
        X = check_array(X, handle_sparse="allow")
        leaves = [self._get_leaf_node(x) for x in X]
        return [self._get_proba_from_leaf_node(x) for x in leaves]
    
    def predict_log_proba(self, X):
        return [list(map(safe_log, x)) for x in self.predict_proba(X)]

class DecisionTreeRegressorPure(_DecisionTreeBase):
    """
    Pure python implementation of `DecisionTreeRegressor`.

    Args:
        estimator (sklearn estimator): fitted `DecisionTreeRegressor` object
    """
    def _get_pred_from_leaf_node(self, node_id):
        return self.value_[node_id][0]

    def predict(self, X):
        X = check_array(X, handle_sparse="allow")
        leaves = [self._get_leaf_node(x) for x in X]
        return [self._get_pred_from_leaf_node(x) for x in leaves]

class ExtraTreeClassifierPure(DecisionTreeClassifierPure):
    """
    Pure python implementation of `ExtraTreeClassifier`.

    Args:
        estimator (sklearn estimator): fitted `ExtraTreeClassifier` object
    """
    pass

class ExtraTreeRegressorPure(DecisionTreeRegressorPure):
    """
    Pure python implementation of `ExtraTreeRegressor`.

    Args:
        estimator (sklearn estimator): fitted `ExtraTreeRegressor` object
    """
    pass
