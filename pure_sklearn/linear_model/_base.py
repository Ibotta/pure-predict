"""
Generalized linear models
"""

from operator import add

from ..utils import check_array, ndim, shape, check_types
from ..base import dot, expit, ravel


class LinearClassifierMixinPure:
    """ Mixin for linear classifiers """

    def __init__(self, estimator):
        self.coef_ = estimator.coef_.tolist()
        self.classes_ = estimator.classes_.tolist()
        if hasattr(estimator, "intercept_"):
            if isinstance(estimator.intercept_, float):
                self.intercept_ = [estimator.intercept_] * len(self.classes_)
            else:
                self.intercept_ = estimator.intercept_.tolist()
        if hasattr(estimator, "multi_class"):
            self.multi_class = estimator.multi_class
        if hasattr(estimator, "solver"):
            self.solver = estimator.solver
        if hasattr(estimator, "loss"):
            self.loss = estimator.loss
        check_types(self)

    def decision_function(self, X):
        """
        Predict confidence scores for samples.
        The confidence score for a sample is the signed distance of that
        sample to the hyperplane.
        """
        X = check_array(X, handle_sparse="allow")

        n_features = shape(self.coef_)[1]
        if shape(X)[1] != n_features:
            raise ValueError(
                "X has %d features per sample; expecting %d" % (shape(X)[1], n_features)
            )

        scores = [
            list(map(add, dot(X[i], self.coef_), self.intercept_))
            for i in range(len(X))
        ]
        return ravel(scores) if shape(scores)[1] == 1 else scores

    def predict(self, X):
        """ Predict class labels for samples in X """
        scores = self.decision_function(X)
        if len(shape(scores)) == 1:
            indices = map(lambda x: int(x > 0), scores)
        else:
            indices = map(lambda a: a.index(max(a)), scores)
        return [self.classes_[i] for i in indices]

    def _predict_proba_lr(self, X):
        """
        Probability estimation for OvR logistic regression.
        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        prob = self.decision_function(X)
        if ndim(prob) == 1:
            return [[1 - a, a] for a in map(expit, prob)]
        else:
            prob = [list(map(expit, a)) for a in prob]
            return [
                list(map(lambda b: (b / sum(a)) if sum(a) != 0 else float("NaN"), a))
                for a in prob
            ]
