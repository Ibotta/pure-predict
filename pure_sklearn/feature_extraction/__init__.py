"""
The :mod:`pure_sklearn.feature_extraction` module deals with feature extraction
from raw data. It currently includes methods to extract features from text.
"""

from ._dict_vectorizer import DictVectorizerPure
from . import text

__all__ = ["DictVectorizerPure", "text"]
