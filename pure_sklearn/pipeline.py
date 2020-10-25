"""
Pipeline and FeatureUnion classes
"""

from operator import add, attrgetter
from functools import update_wrapper, reduce
from itertools import islice

from .base import accumu, apply_2d
from .utils import (
    issparse,
    tosparse,
    shape,
    check_array,
    check_types,
    sparse_list,
    check_version,
)
from .map import convert_estimator

__all__ = ["FeatureUnionPure", "PipelinePure"]


class _IffHasAttrDescriptor:
    """ Implements a conditional property using the descriptor protocol """

    def __init__(self, fn, delegate_names, attribute_name):
        self.fn = fn
        self.delegate_names = delegate_names
        self.attribute_name = attribute_name
        update_wrapper(self, fn)

    def __get__(self, obj, type=None):
        if obj is not None:
            for delegate_name in self.delegate_names:
                try:
                    delegate = attrgetter(delegate_name)(obj)
                except AttributeError:
                    continue
                else:
                    getattr(delegate, self.attribute_name)
                    break
            else:
                attrgetter(self.delegate_names[-1])(obj)

        out = lambda *args, **kwargs: self.fn(obj, *args, **kwargs)
        update_wrapper(out, self.fn)
        return out


def _if_delegate_has_method(delegate):
    if isinstance(delegate, list):
        delegate = tuple(delegate)
    if not isinstance(delegate, tuple):
        delegate = (delegate,)
    return lambda fn: _IffHasAttrDescriptor(fn, delegate, attribute_name=fn.__name__)


def _transform_one(transformer, X, y, weight, **fit_params):
    res = transformer.transform(X)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res
    return apply_2d(res, lambda x: x * weight)


class PipelinePure:
    """
    Pure python implementation of `Pipeline`.

    Args:
        estimator (sklearn estimator): fitted `Pipeline` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        self.steps = []
        for step in estimator.steps:
            step_ = convert_estimator(step[1])
            self.steps.append((step[0], step_))
        check_types(self)

    def _iter(self, with_final=True, filter_passthrough=True):
        """
        Generate (idx, (name, trans)) tuples from self.steps
        When filter_passthrough is True, 'passthrough' and None transformers
        are filtered out.
        """
        stop = len(self.steps)
        if not with_final:
            stop -= 1

        for idx, (name, trans) in enumerate(islice(self.steps, 0, stop)):
            if not filter_passthrough:
                yield idx, name, trans
            elif trans is not None and trans != "passthrough":
                yield idx, name, trans

    def __len__(self):
        """ Returns the length of the Pipeline """
        return len(self.steps)

    @_if_delegate_has_method(delegate="_final_estimator")
    def predict(self, X, **predict_params):
        """ Apply transforms to the data, and predict with the final estimator """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict(Xt, **predict_params)

    @_if_delegate_has_method(delegate="_final_estimator")
    def predict_proba(self, X):
        """ Apply transforms, and predict_proba of the final estimator """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_proba(Xt)

    @_if_delegate_has_method(delegate="_final_estimator")
    def predict_log_proba(self, X):
        """ Apply transforms, and predict_log_proba of the final estimator """
        Xt = X
        for _, name, transform in self._iter(with_final=False):
            Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_log_proba(Xt)

    @property
    def transform(self):
        """
        Apply transforms, and transform with the final estimator
        This also works where final estimator is ``None``: all prior
        transformations are applied.
        """
        if self._final_estimator != "passthrough":
            self._final_estimator.transform
        return self._transform

    def _transform(self, X):
        Xt = X
        for _, _, transform in self._iter():
            Xt = transform.transform(Xt)
        return Xt

    @property
    def classes_(self):
        return self.steps[-1][-1].classes_

    @property
    def _final_estimator(self):
        estimator = self.steps[-1][1]
        return "passthrough" if estimator is None else estimator


class FeatureUnionPure:
    """
    Pure python implementation of `FeatureUnion`.

    Args:
        estimator (sklearn estimator): fitted `FeatureUnion` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        self.transformer_list = []
        for step in estimator.transformer_list:
            step_ = convert_estimator(step[1])
            self.transformer_list.append((step[0], step_))

        if hasattr(estimator, "transformer_weights"):
            if estimator.transformer_weights is None:
                self.transformer_weights = estimator.transformer_weights
            else:
                self.transformer_weights = {}
                for k, v in estimator.transformer_weights.items():
                    self.transformer_weights[k] = float(v)
        check_types(self)

    def _iter(self):
        get_weight = (self.transformer_weights or {}).get
        return (
            (name, trans, get_weight(name))
            for name, trans in self.transformer_list
            if trans is not None and trans != "drop"
        )

    def transform(self, X):
        """
        Transform X separately by each transformer
        and concatenate results.
        """
        X = check_array(X, handle_sparse="allow")
        Xs = [
            _transform_one(trans, X, None, weight)
            for name, trans, weight in self._iter()
        ]
        if not Xs:
            return [[0.0] * shape(X)[1]] * shape(X)[0]

        if any(issparse(f) for f in Xs):
            Xs = [tosparse(X_) if not issparse(X_) else X_ for X_ in Xs]
            sizes = [x.size for x in Xs]
            start_indices = [0] + list(accumu(sizes))

            # concatenate dictionaries for each row
            # appropriately updating indices by a cumulative
            # sum of sparse X sizes
            func = lambda index: [
                list(
                    {(k + start_indices[i]): v for k, v in Xs[i][index].items()}.items()
                )
                for i in range(len(Xs))
            ]
            X_total = [dict(reduce(add, func(index))) for index in range(len(X))]
            return sparse_list(X_total, size=sum(sizes), dtype=float)
        else:
            return [
                list(reduce(add, [X_[index] for X_ in Xs])) for index in range(len(X))
            ]
