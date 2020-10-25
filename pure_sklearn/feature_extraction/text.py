"""
The :mod:`pure_sklearn.feature_extraction.text` submodule gathers utilities to
build feature vectors from text documents.
"""

import re
import unicodedata

from functools import partial
from collections import defaultdict
from math import isnan

from ..utils import (
    check_types,
    convert_type,
    sparse_list,
    shape,
    check_array,
    check_types,
    check_version,
)
from ..map import convert_estimator
from ..preprocessing import normalize_pure
from ._hash import _FeatureHasherPure

__all__ = [
    "CountVectorizerPure",
    "TfidfTransformerPure",
    "TfidfVectorizerPure",
    "HashingVectorizerPure",
]


def _preprocess(doc, accent_function=None, lower=False):
    """
    Chain together an optional series of text preprocessing steps to
    apply to a document.
    """
    if lower:
        doc = doc.lower()
    if accent_function is not None:
        doc = accent_function(doc)
    return doc


def _analyze(
    doc,
    analyzer=None,
    tokenizer=None,
    ngrams=None,
    preprocessor=None,
    decoder=None,
    stop_words=None,
):
    """
    Chain together an optional series of text processing steps to go from
    a single document to ngrams, with or without tokenizing or preprocessing.
    """

    if decoder is not None:
        doc = decoder(doc)
    if analyzer is not None:
        doc = analyzer(doc)
    else:
        if preprocessor is not None:
            doc = preprocessor(doc)
        if tokenizer is not None:
            doc = tokenizer(doc)
        if ngrams is not None:
            if stop_words is not None:
                doc = ngrams(doc, stop_words)
            else:
                doc = ngrams(doc)
    return doc


def strip_accents_unicode(s):
    """ Transform accentuated unicode symbols into their simple counterpart """
    try:
        # If `s` is ASCII-compatible, then it does not contain any accented
        # characters and we can avoid an expensive list comprehension
        s.encode("ASCII", errors="strict")
        return s
    except UnicodeEncodeError:
        normalized = unicodedata.normalize("NFKD", s)
        return "".join([c for c in normalized if not unicodedata.combining(c)])


def strip_accents_ascii(s):
    """ Transform accentuated unicode symbols into ascii or nothing """
    nkfd_form = unicodedata.normalize("NFKD", s)
    return nkfd_form.encode("ASCII", "ignore").decode("ASCII")


def strip_tags(s):
    """ Basic regexp based HTML / XML tag stripper function """
    return re.compile(r"<([^>]+)>", flags=re.UNICODE).sub(" ", s)


def _check_stop_list(stop):
    if stop == "english":
        raise ValueError(
            "English stopwords not supported. Pass explicitly as a custom stopwords list."
        )
    elif isinstance(stop, str):
        raise ValueError("not a built-in stop list: %s" % stop)
    elif stop is None:
        return None
    else:  # assume it's a collection
        return frozenset(stop)


class _VectorizerMixinPure:
    """ Provides common code for text vectorizers (tokenization logic) """

    _white_spaces = re.compile(r"\s\s+")

    def decode(self, doc):
        """ Decode the input into a string of unicode symbols """
        if self.input == "filename":
            with open(doc, "rb") as fh:
                doc = fh.read()

        elif self.input == "file":
            doc = doc.read()

        if isinstance(doc, bytes):
            doc = doc.decode(self.encoding, self.decode_error)

        if not isinstance(doc, str) and isnan(doc):
            raise ValueError(
                "np.nan is an invalid document, expected byte or unicode string."
            )

        return doc

    def _word_ngrams(self, tokens, stop_words=None):
        """ Turn tokens into a sequence of n-grams after stop words filtering """
        # handle stop words
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i : i + n]))

        return tokens

    def _char_ngrams(self, text_document):
        """ Tokenize text_document into a sequence of character n-grams """
        # normalize white spaces
        text_document = self._white_spaces.sub(" ", text_document)

        text_len = len(text_document)
        min_n, max_n = self.ngram_range
        if min_n == 1:
            # no need to do any slicing for unigrams
            # iterate through the string
            ngrams = list(text_document)
            min_n += 1
        else:
            ngrams = []

        # bind method outside of loop to reduce overhead
        ngrams_append = ngrams.append

        for n in range(min_n, min(max_n + 1, text_len + 1)):
            for i in range(text_len - n + 1):
                ngrams_append(text_document[i : i + n])
        return ngrams

    def _char_wb_ngrams(self, text_document):
        """ Whitespace sensitive char-n-gram tokenization """
        # normalize white spaces
        text_document = self._white_spaces.sub(" ", text_document)

        min_n, max_n = self.ngram_range
        ngrams = []

        # bind method outside of loop to reduce overhead
        ngrams_append = ngrams.append

        for w in text_document.split():
            w = " " + w + " "
            w_len = len(w)
            for n in range(min_n, max_n + 1):
                offset = 0
                ngrams_append(w[offset : offset + n])
                while offset + n < w_len:
                    offset += 1
                    ngrams_append(w[offset : offset + n])
                if offset == 0:  # count a short word (w_len < n) only once
                    break
        return ngrams

    def build_preprocessor(self):
        """ Return a function to preprocess the text before tokenization """
        if self.preprocessor is not None:
            return self.preprocessor

        # accent stripping
        if not self.strip_accents:
            strip_accents = None
        elif callable(self.strip_accents):
            strip_accents = self.strip_accents
        elif self.strip_accents == "ascii":
            strip_accents = strip_accents_ascii
        elif self.strip_accents == "unicode":
            strip_accents = strip_accents_unicode
        else:
            raise ValueError(
                'Invalid value for "strip_accents": %s' % self.strip_accents
            )

        return partial(_preprocess, accent_function=strip_accents, lower=self.lowercase)

    def build_tokenizer(self):
        """ Return a function that splits a string into a sequence of tokens """
        if self.tokenizer is not None:
            return self.tokenizer
        token_pattern = re.compile(self.token_pattern)
        return token_pattern.findall

    def get_stop_words(self):
        """ Build or fetch the effective stop words list """
        return _check_stop_list(self.stop_words)

    def _check_stop_words_consistency(self, stop_words, preprocess, tokenize):
        """ Check if stop words are consistent """
        if id(self.stop_words) == getattr(self, "_stop_words_id", None):
            # Stop words are were previously validated
            return None

        # NB: stop_words is validated, unlike self.stop_words
        try:
            inconsistent = set()
            for w in stop_words or ():
                tokens = list(tokenize(preprocess(w)))
                for token in tokens:
                    if token not in stop_words:
                        inconsistent.add(token)
            self._stop_words_id = id(self.stop_words)

            if inconsistent:
                warnings.warn(
                    "Your stop_words may be inconsistent with "
                    "your preprocessing. Tokenizing the stop "
                    "words generated tokens %r not in "
                    "stop_words." % sorted(inconsistent)
                )
            return not inconsistent
        except Exception:
            # Failed to check stop words consistency (e.g. because a custom
            # preprocessor or tokenizer was used)
            self._stop_words_id = id(self.stop_words)
            return "error"

    def build_analyzer(self):
        """
        Return a callable that handles preprocessing, tokenization
        and n-grams generation.
        """

        if callable(self.analyzer):
            if self.input in ["file", "filename"]:
                self._validate_custom_analyzer()
            return partial(_analyze, analyzer=self.analyzer, decoder=self.decode)

        preprocess = self.build_preprocessor()

        if self.analyzer == "char":
            return partial(
                _analyze,
                ngrams=self._char_ngrams,
                preprocessor=preprocess,
                decoder=self.decode,
            )

        elif self.analyzer == "char_wb":

            return partial(
                _analyze,
                ngrams=self._char_wb_ngrams,
                preprocessor=preprocess,
                decoder=self.decode,
            )

        elif self.analyzer == "word":
            stop_words = self.get_stop_words()
            tokenize = self.build_tokenizer()
            self._check_stop_words_consistency(stop_words, preprocess, tokenize)
            return partial(
                _analyze,
                ngrams=self._word_ngrams,
                tokenizer=tokenize,
                preprocessor=preprocess,
                decoder=self.decode,
                stop_words=stop_words,
            )

        else:
            raise ValueError(
                "%s is not a valid tokenization scheme/analyzer" % self.analyzer
            )


class CountVectorizerPure(_VectorizerMixinPure):
    """
    Pure python implementation of `CountVectorizer`.

    Args:
        estimator (sklearn estimator): fitted `CountVectorizer` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        self.dtype = convert_type(estimator.dtype)
        self.binary = estimator.binary
        self.vocabulary_ = {k: int(v) for k, v in estimator.vocabulary_.items()}
        self.analyzer = estimator.analyzer
        self.preprocessor = estimator.preprocessor
        self.tokenizer = estimator.tokenizer
        self.stop_words = estimator.stop_words
        self.token_pattern = estimator.token_pattern
        self.ngram_range = estimator.ngram_range
        self.strip_accents = estimator.strip_accents
        self.decode_error = estimator.decode_error
        self.encoding = estimator.encoding
        self.lowercase = estimator.lowercase
        self.input = estimator.input
        check_types(self)

    def _count_vocab(self, raw_documents):
        """ Create sparse feature matrix, and vocabulary where fixed_vocab=False """
        vocabulary = self.vocabulary_
        analyze = self.build_analyzer()
        data = []
        for doc in raw_documents:
            feature_counter = {}
            for feature in analyze(doc):
                try:
                    feature_idx = vocabulary[feature]
                    if feature_idx not in feature_counter:
                        feature_counter[feature_idx] = 1
                    else:
                        feature_counter[feature_idx] += 1
                except KeyError:
                    continue
            data.append(feature_counter)
        X = sparse_list(data, size=len(vocabulary), dtype=self.dtype)
        return vocabulary, X

    def transform(self, raw_documents):
        """ Transform documents to document-term matrix """
        if isinstance(raw_documents, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )

        _, X = self._count_vocab(raw_documents)
        if self.binary:
            X = [dict.fromkeys(x, 1) for x in X]
        return X


class TfidfVectorizerPure(CountVectorizerPure):
    """
    Pure python implementation of `TfidfVectorizer`.

    Args:
        estimator (sklearn estimator): fitted `TfidfVectorizer` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        self._tfidf = convert_estimator(estimator._tfidf)
        super().__init__(estimator)

    def transform(self, raw_documents):
        """Transform documents to document-term matrix."""
        X = super().transform(raw_documents)
        return self._tfidf.transform(X, copy=False)


class TfidfTransformerPure:
    """
    Pure python implementation of `TfidfTransformer`.

    Args:
        estimator (sklearn estimator): fitted `TfidfTransformer` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        self.norm = estimator.norm
        self.use_idf = estimator.use_idf
        self.smooth_idf = estimator.smooth_idf
        self.sublinear_tf = estimator.sublinear_tf
        self.idf_ = estimator.idf_.tolist()
        self.expected_n_features_ = estimator._idf_diag.shape[0]
        check_types(self)

    def transform(self, X, copy=True):
        X = check_array(X, handle_sparse="allow")
        n_samples, n_features = shape(X)

        if self.sublinear_tf:
            for index in range(len(X)):
                X[index] = safe_log(X[index]) + 1

        if self.use_idf:
            if n_features != self.expected_n_features_:
                raise ValueError(
                    "Input has n_features=%d while the model"
                    " has been trained with n_features=%d"
                    % (n_features, expected_n_features)
                )
            for index in range(len(X)):
                for k, v in X[index].items():
                    X[index][k] = v * self.idf_[k]

        if self.norm:
            X = normalize_pure(X, norm=self.norm, copy=False)

        return X


class HashingVectorizerPure(_VectorizerMixinPure):
    """
    Pure python implementation of `HashingVectorizer`.

    Args:
        estimator (sklearn estimator): fitted `HashingVectorizer` object
    """

    def __init__(self, estimator):
        check_version(estimator)
        self.dtype = convert_type(estimator.dtype)
        self.norm = estimator.norm
        self.binary = estimator.binary
        self.analyzer = estimator.analyzer
        self.preprocessor = estimator.preprocessor
        self.tokenizer = estimator.tokenizer
        self.stop_words = estimator.stop_words
        self.token_pattern = estimator.token_pattern
        self.ngram_range = estimator.ngram_range
        self.strip_accents = estimator.strip_accents
        self.decode_error = estimator.decode_error
        self.encoding = estimator.encoding
        self.lowercase = estimator.lowercase
        self.input = estimator.input
        self.n_features = estimator.n_features
        self.alternate_sign = estimator.alternate_sign
        check_types(self)

    def transform(self, X):
        """ Transform a sequence of documents to a document-term matrix """
        if isinstance(X, str):
            raise ValueError(
                "Iterable over raw text documents expected, string object received."
            )

        analyzer = self.build_analyzer()
        X = self._get_hasher().transform(analyzer(doc) for doc in X)
        if self.binary:
            X = [dict.fromkeys(x, 1) for x in X]

        if self.norm is not None:
            X = normalize_pure(X, norm=self.norm, copy=False)

        return X

    def _get_hasher(self):
        return _FeatureHasherPure(
            n_features=self.n_features,
            input_type="string",
            dtype=self.dtype,
            alternate_sign=self.alternate_sign,
        )
