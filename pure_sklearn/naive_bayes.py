"""
Naive Bayes 
"""

from abc import abstractmethod
from math import pi

from .base import dot, transpose, safe_log, safe_exp
from .utils import check_array, check_types, check_version

__all__ = ["GaussianNBPure", "MultinomialNBPure", "ComplementNBPure"]


class _BaseNBPure:
    """ Base class for naive Bayes classifiers """

    @abstractmethod
    def _joint_log_likelihood(self, X):
        """ Compute the unnormalized posterior log probability of X """

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.
        """
        X = check_array(X, handle_sparse="error")
        jll = self._joint_log_likelihood(X)
        indices = map(lambda a: a.index(max(a)), jll)
        return [self.classes_[i] for i in indices]

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.
        """
        X = check_array(X, handle_sparse="error")
        jll = self._joint_log_likelihood(X)
        log_prob_x = list(map(lambda a: safe_log(sum(map(safe_exp, a))), jll))
        return [
            list(map(lambda a: a - log_prob_x[index], jll[index]))
            for index in range(len(jll))
        ]

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.
        """
        return [list(map(safe_exp, a)) for a in self.predict_log_proba(X)]


class GaussianNBPure(_BaseNBPure):
    """
    Pure python implementation of `GaussianNB`.

    Args:
        estimator (sklearn estimator): fitted `GaussianNB` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        self.class_prior_ = estimator.class_prior_.tolist()
        self.classes_ = estimator.classes_.tolist()
        self.sigma_ = estimator.sigma_.tolist()
        self.theta_ = estimator.theta_.tolist()
        check_types(self)

    def _joint_log_likelihood(self, X):
        """ Calculate the posterior log probability of the samples X """
        joint_log_likelihood = []
        for i in range(len(self.classes_)):
            jointi = safe_log(self.class_prior_[i])
            n_ij = -0.5 * sum(
                list(map(lambda x: safe_log(2.0 * pi * x), self.sigma_[i]))
            )
            jll = [
                list(
                    map(
                        lambda b: ((a[b] - self.theta_[i][b]) ** 2) / self.sigma_[i][b],
                        range(len(a)),
                    )
                )
                for a in X
            ]
            jll = list(map(lambda a: 0.5 * sum(a), jll))
            jll = [(n_ij - a) + jointi for a in jll]
            joint_log_likelihood.append(jll)
        return transpose(joint_log_likelihood)


class MultinomialNBPure(_BaseNBPure):
    """
    Pure python implementation of `MultinomialNB`.

    Args:
        estimator (sklearn estimator): fitted `MultinomialNB` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        self.class_log_prior_ = estimator.class_log_prior_.tolist()
        self.classes_ = estimator.classes_.tolist()
        self.feature_log_prob_ = estimator.feature_log_prob_.tolist()
        check_types(self)

    def _joint_log_likelihood(self, X):
        """ Calculate the posterior log probability of the samples X """
        return [self._jll(a) for a in X]

    def _jll(self, x):
        """ Calculate the joint log likelihood for one sample """
        dot_prod = dot(x, self.feature_log_prob_)
        return [
            (dot_prod[index] + self.class_log_prior_[index])
            for index in range(len(self.classes_))
        ]


class ComplementNBPure(_BaseNBPure):
    """
    Pure python implementation of `ComplementNB`.

    Args:
        estimator (sklearn estimator): fitted `ComplementNB` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        self.class_log_prior_ = estimator.class_log_prior_.tolist()
        self.classes_ = estimator.classes_.tolist()
        self.feature_log_prob_ = estimator.feature_log_prob_.tolist()
        check_types(self)

    def _joint_log_likelihood(self, X):
        """ Calculate the class scores for the samples in X """
        jll = [dot(x, self.feature_log_prob_) for x in X]
        if len(self.classes_) == 1:
            jll = [[x[0] + self.class_log_prior_[0]] for x in jll]
        return jll
