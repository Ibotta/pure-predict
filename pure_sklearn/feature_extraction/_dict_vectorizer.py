"""
Dictionary vectorizer
"""

from collections.abc import Mapping

from ..utils import check_types, sparse_list, convert_type, check_version


class DictVectorizerPure:
    """
    Pure python implementation of `DictVectorizer`.

    Args:
        estimator (sklearn estimator): fitted `DictVectorizer` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        self.vocabulary_ = estimator.vocabulary_
        self.feature_names_ = estimator.feature_names_
        self.sparse = estimator.sparse
        self.dtype = convert_type(estimator.dtype)
        self.separator = estimator.separator
        check_types(self)

    def transform(self, X):
        dtype = self.dtype
        feature_names = self.feature_names_
        vocab = self.vocabulary_
        X = [X] if isinstance(X, Mapping) else X

        data = []
        for x in X:
            row = {}
            for f, v in x.items():
                if isinstance(v, str):
                    f = "%s%s%s" % (f, self.separator, v)
                    v = 1
                if f in vocab:
                    row[vocab[f]] = dtype(v)
            data.append(row)
        result_matrix = sparse_list(data, size=len(vocab), dtype=dtype)
        if not self.sparse:
            result_matrix = result_matrix.todense()
        return result_matrix
