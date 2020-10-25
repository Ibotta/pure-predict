"""
Utility functions
"""

import pickle
import time

from warnings import warn
from distutils.version import LooseVersion

CONTAINERS = (list, dict, tuple)
TYPES = (int, float, str, bool, type)
MIN_VERSION = "0.20"


def check_types(obj, containers=CONTAINERS, types=TYPES):
    """
    Checks if input object is an allowed type. Objects can be
    acceptable containers or acceptable types themselves.
    Containers are checked recursively to ensure all contained
    types are valid. If object is a `pure_sklearn` type, its
    attributes are all recursively checked.
    """
    if isinstance(obj, containers):
        if isinstance(obj, (list, tuple)):
            for ob in obj:
                check_types(ob)
        else:
            for k, v in obj.items():
                check_types(k)
                check_types(v)
    elif isinstance(obj, types):
        pass
    elif "pure_sklearn" in str(type(obj)):
        for attr in vars(obj):
            check_types(getattr(obj, attr))
    elif obj is None:
        pass
    else:
        raise ValueError("Object contains invalid type: {}".format(type(obj)))


def check_version(estimator, min_version=None):
    """ Checks the version of the scikit-learn estimator """
    warning_str = (
        "Estimators fitted with sklearn version < {} are not guaranteed to work".format(
            MIN_VERSION
        )
    )
    try:
        version_ = estimator.__getstate__()["_sklearn_version"]
    except:
        warn(warning_str)
        return
    if (min_version is not None) and (
        LooseVersion(version_) < LooseVersion(min_version)
    ):
        raise Exception(
            "The sklearn version is too low for this estimator; must be >= {}".format(
                min_version
            )
        )
    elif LooseVersion(version_) < LooseVersion(MIN_VERSION):
        warn(warning_str)


def convert_type(dtype):
    """ Converts a datatype to its pure python equivalent """
    val = dtype(0)
    if hasattr(val, "item"):
        return type(val.item())
    else:
        return dtype


def check_array(X, handle_sparse="error"):
    """
    Checks if array is compatible for prediction with
    `pure_sklearn` classes. Input 'X' should be a non-empty
    `list` or `sparse_list`. If 'X' is sparse, flexible
    sparse handling is applied, allowing sparse by default,
    or optionally erroring on sparse input.
    """
    if issparse(X):
        if handle_sparse == "allow":
            return X
        elif handle_sparse == "error":
            raise ValueError("Sparse input is not supported " "for this estimator")
        else:
            raise ValueError(
                "Invalid value for 'handle_sparse' "
                "input. Acceptable values are 'allow' or 'error'"
            )
    if not isinstance(X, list):
        raise TypeError("Input 'X' must be a list")
    if len(X) == 0:
        return ValueError("Input 'X' must not be empty")
    return X


def shape(X):
    """
    Checks the shape of input list. Similar to
    numpy `ndarray.shape()`. Handles `list` or
    `sparse_list` input.
    """
    if ndim(X) == 1:
        return (len(X),)
    elif ndim(X) == 2:
        if issparse(X):
            return (len(X), X.size)
        else:
            return (len(X), len(X[0]))


def ndim(X):
    """ Computes the dimension of input list """
    if isinstance(X[0], (list, dict)):
        return 2
    else:
        return 1


def tosparse(A):
    """ Converts input dense list to a `sparse_list` """
    return sparse_list(A)


def todense(A):
    """ Converts input `sparse_list` to a dense list """
    return A.todense()


def issparse(A):
    """ Checks if input list is a `sparse_list` """
    return isinstance(A, sparse_list)


class sparse_list(list):
    """
    Pure python implementation of a 2-D sparse data structure.
    The data structure is a list of dictionaries. Each dictionary
    represents a 'row' of data. The dictionary keys correspond to the
    indices of 'columns' and the dictionary values correspond to the
    data value associated with that index. Missing keys are assumed
    to have values of 0.

    Args:
        A (list): 2-D list of lists or list of dicts
        size (int): Number of 'columns' of the data structure
        dtype (type): Data type of data values

    Examples:
    >>> A = [[0,1,0], [0,1,1]]
    >>> print(sparse_list(A))
    ... [{1:1}, {2:1, 3:1}]
    >>>
    >>> B = [{3:0.5}, {1:0.9, 10:0.2}]
    >>> print(sparse_list(B, size=11, dtype=float))
    ... [{3:0.5}, {1:0.9, 10:0.2}]
    """

    def __init__(self, A, size=None, dtype=None):
        if isinstance(A[0], dict):
            self.dtype = float if dtype is None else dtype
            self.size = size
            for row in A:
                self.append(row)
        else:
            A = check_array(A)
            self.size = shape(A)[1]
            self.dtype = type(A[0][0])
            for row in A:
                self.append(
                    dict([(i, row[i]) for i in range(self.size) if row[i] != 0])
                )

    def todense(self):
        """ Converts `sparse_list` instance to a dense list """
        A_dense = []
        zero_val = self.dtype(0)
        for row in self:
            A_dense.append([row.get(i, zero_val) for i in range(self.size)])
        return A_dense


def performance_comparison(sklearn_estimator, pure_sklearn_estimator, X):
    """
    Profile performance characteristics between sklearn estimator and
    corresponding pure-predict estimator.

    Args:
        sklearn_estimator (object)
        pure_sklearn_estimator (object)
        X (numpy ndarray): features for prediction
    """
    ### -- profile pickled object size: sklearn vs pure-predict
    pickled = pickle.dumps(sklearn_estimator)
    pickled_ = pickle.dumps(pure_sklearn_estimator)
    print("Pickle Size sklearn: {}".format(len(pickled)))
    print("Pickle Size pure-predict: {}".format(len(pickled_)))
    print("Difference: {}".format(len(pickled_) / float(len(pickled))))

    ### -- profile unpickle time: sklearn vs pure-predict
    start = time.time()
    _ = pickle.loads(pickled)
    pickle_t = time.time() - start
    print("Unpickle time sklearn: {}".format(pickle_t))
    start = time.time()
    _ = pickle.loads(pickled_)
    pickle_t_ = time.time() - start
    print("Unpickle time pure-predict: {}".format(pickle_t_))
    print("Difference: {}".format(pickle_t_ / pickle_t))

    ### -- profile single record predict latency: sklearn vs pure-predict
    X_pred = X[:1]
    X_pred_ = X_pred if isinstance(X_pred, list) else X_pred.tolist()
    start = time.time()
    _ = sklearn_estimator.predict(X_pred)
    pred_t = time.time() - start
    print("Predict 1 record sklearn: {}".format(pred_t))
    start = time.time()
    _ = pure_sklearn_estimator.predict(X_pred_)
    pred_t_ = time.time() - start
    print("Predict 1 record pure-predict: {}".format(pred_t_))
    print("Difference: {}".format(pred_t_ / pred_t))
