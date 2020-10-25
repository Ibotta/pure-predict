"""
Feature hashing

Pure python implementation of murmur3 hash written here: https://github.com/wc-duck/pymmh3
and submitted to the public domain. It has been replicated here in order to reduce 
external dependencies. 
"""

import numbers

from ..utils import check_types, sparse_list

MAX_INT = 2147483647


def _xrange(a, b, c):
    return range(a, b, c)


def _xencode(x):
    if isinstance(x, (bytes, bytearray)):
        return x
    else:
        return x.encode()


def _hash(key, seed=0x0):
    """ Implements 32bit murmur3 hash """

    key = bytearray(_xencode(key))

    def fmix(h):
        h ^= h >> 16
        h = (h * 0x85EBCA6B) & 0xFFFFFFFF
        h ^= h >> 13
        h = (h * 0xC2B2AE35) & 0xFFFFFFFF
        h ^= h >> 16
        return h

    length = len(key)
    nblocks = int(length / 4)

    h1 = seed

    c1 = 0xCC9E2D51
    c2 = 0x1B873593

    # body
    for block_start in _xrange(0, nblocks * 4, 4):
        # ??? big endian?
        k1 = (
            key[block_start + 3] << 24
            | key[block_start + 2] << 16
            | key[block_start + 1] << 8
            | key[block_start + 0]
        )

        k1 = (c1 * k1) & 0xFFFFFFFF
        k1 = (k1 << 15 | k1 >> 17) & 0xFFFFFFFF  # inlined ROTL32
        k1 = (c2 * k1) & 0xFFFFFFFF

        h1 ^= k1
        h1 = (h1 << 13 | h1 >> 19) & 0xFFFFFFFF  # inlined ROTL32
        h1 = (h1 * 5 + 0xE6546B64) & 0xFFFFFFFF

    # tail
    tail_index = nblocks * 4
    k1 = 0
    tail_size = length & 3

    if tail_size >= 3:
        k1 ^= key[tail_index + 2] << 16
    if tail_size >= 2:
        k1 ^= key[tail_index + 1] << 8
    if tail_size >= 1:
        k1 ^= key[tail_index + 0]

    if tail_size > 0:
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = (k1 << 15 | k1 >> 17) & 0xFFFFFFFF  # inlined ROTL32
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1

    # finalization
    unsigned_val = fmix(h1 ^ length)
    if unsigned_val & 0x80000000 == 0:
        return unsigned_val
    else:
        return -((unsigned_val ^ 0xFFFFFFFF) + 1)


def _hashing_transform(raw_X, n_features, dtype, alternate_sign=1, seed=0):
    """ Guts of FeatureHasher.transform """
    assert n_features > 0
    X = []
    for x in raw_X:
        row = {}
        for f, v in x:
            if isinstance(v, str):
                f = "%s%s%s" % (f, "=", v)
                value = 1
            else:
                value = v

            if value == 0:
                continue

            h = _hash(f, seed)
            index = abs(h) % n_features
            if alternate_sign:
                value *= (h >= 0) * 2 - 1
            row[index] = value
        X.append(row)
    return sparse_list(X, size=n_features, dtype=dtype)


class _FeatureHasherPure:
    """ Pure python implementation of `FeatureHasher` """

    def __init__(
        self, n_features=(2 ** 20), input_type="dict", dtype=float, alternate_sign=True
    ):
        self._validate_params(n_features, input_type)
        self.dtype = dtype
        self.input_type = input_type
        self.n_features = n_features
        self.alternate_sign = alternate_sign
        check_types(self)

    @staticmethod
    def _validate_params(n_features, input_type):
        if not isinstance(n_features, numbers.Integral):
            raise TypeError(
                "n_features must be integral, got %r (%s)."
                % (n_features, type(n_features))
            )
        elif n_features < 1 or n_features >= MAX_INT + 1:
            raise ValueError("Invalid number of features (%d)." % n_features)

        if input_type not in ("dict", "pair", "string"):
            raise ValueError(
                "input_type must be 'dict', 'pair' or 'string', got %r." % input_type
            )

    def transform(self, raw_X):
        """ Transform a sequence of instances to a `sparse_list` """
        raw_X = iter(raw_X)
        if self.input_type == "dict":
            raw_X = (_iteritems(d) for d in raw_X)
        elif self.input_type == "string":
            raw_X = (((f, 1) for f in x) for x in raw_X)
        return _hashing_transform(
            raw_X, self.n_features, self.dtype, self.alternate_sign, seed=0
        )
