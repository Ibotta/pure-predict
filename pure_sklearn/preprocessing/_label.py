"""
Label helper functions
"""


def _encode_python(values, uniques=None, encode=False):
    # only used in _encode below, see docstring there for details
    if uniques is None:
        uniques = list(sorted(set(values)))
    if encode:
        table = {val: i for i, val in enumerate(uniques)}
        try:
            encoded = [table[v] for v in values]
        except KeyError as e:
            raise ValueError("y contains previously unseen labels: %s" % str(e))
        return uniques, encoded
    else:
        return uniques


def _encode(values, uniques=None, encode=False, check_unknown=True):
    """ Helper function to factorize (find uniques) and encode values """
    try:
        res = _encode_python(values, uniques, encode)
    except TypeError:
        raise TypeError("argument must be a string or number")
    return res


def _encode_check_unknown(values, uniques, return_mask=False):
    """ Helper function to check for unknowns in values to be encoded """
    uniques_set = set(uniques)
    diff = list(set(values) - uniques_set)
    if return_mask:
        if diff:
            valid_mask = [val in uniques_set for val in values]
        else:
            valid_mask = [True] * len(values)
        return diff, valid_mask
    else:
        return diff
