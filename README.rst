.. figure:: https://github.com/Ibotta/pure-predict/blob/master/doc/images/pure-predict.png
   :alt: pure-predict

pure-predict: Machine learning prediction in pure Python
========================================================

|License| |Build Status| |PyPI Package| |Downloads| |Python Versions|

``pure-predict`` speeds up and slims down machine learning prediction applications. It is 
a foundational tool for serverless inference or small batch prediction with popular machine 
learning frameworks like `scikit-learn <https://scikit-learn.org/stable/>`__ and `fasttext <https://fasttext.cc/>`__. 
It implements the predict methods of these frameworks in pure Python.

Primary Use Cases
-----------------
The primary use case for ``pure-predict`` is the following scenario: 

#. A model is trained in an environment without strong container footprint constraints. Perhaps a long running "offline" job on one or many machines where installing a number of python packages from PyPI is not at all problematic.
#. At prediction time the model needs to be served behind an API. Typical access patterns are to request a prediction for one "record" (one "row" in a ``numpy`` array or one string of text to classify) per request or a mini-batch of records per request.
#. Preferred infrastructure for the prediction service is either serverless (`AWS Lambda <https://aws.amazon.com/lambda/>`__) or a container service where the memory footprint of the container is constrained.
#. The fitted model object's artifacts needed for prediction (coefficients, weights, vocabulary, decision tree artifacts, etc.) are relatively small (10s to 100s of MBs).

.. figure:: https://github.com/Ibotta/pure-predict/blob/master/doc/images/diagram.png
   :alt: diagram

In this scenario, a container service with a large dependency footprint can be overkill for a microservice, particularly if the access patterns favor the pricing model of a serverless application. Additionally, for smaller models and single record predictions per request, the ``numpy`` and ``scipy`` functionality in the prediction methods of popular machine learning frameworks work against the application in terms of latency, `underperforming pure python <https://github.com/Ibotta/pure-predict/blob/master/examples/performance_rf.py>`__ in some cases.

Check out the `blog post <https://medium.com/building-ibotta/predict-with-sklearn-20x-faster-9f2803944446>`__ 
for more information on the motivation and use cases of ``pure-predict``.

Package Details
---------------

It is a Python package for machine learning prediction distributed under 
the `Apache 2.0 software license <https://github.com/Ibotta/sk-dist/blob/master/LICENSE>`__. 
It contains multiple subpackages which mirror their open source 
counterpart (``scikit-learn``, ``fasttext``, etc.). Each subpackage has utilities to 
convert a fitted machine learning model into a custom object containing prediction methods 
that mirror their native counterparts, but converted to pure python. Additionally, all 
relevant model artifacts needed for prediction are converted to pure python. 

A ``pure-predict`` model object can then be pickled and later
unpickled without any 3rd party dependencies other than ``pure-predict``.

This eliminates the need to have large dependency packages installed in order to 
make predictions with fitted machine learning models using popular open source packages for
training models. These dependencies (``numpy``, ``scipy``, ``scikit-learn``, ``fasttext``, etc.) 
are large in size and `not always necessary to make fast and accurate
predictions <https://github.com/Ibotta/pure-predict/blob/master/examples/performance_rf.py>`__. 
Additionally, they rely on C extensions that may not be ideal for serverless applications with a python runtime.

Quick Start Example
-------------------

In a python enviornment with ``scikit-learn`` and its dependencies installed:

.. code-block:: python
    
    import pickle
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from pure_sklearn.map import convert_estimator
    
    # fit sklearn estimator
    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier()
    clf.fit(X, y)
    
    # convert to pure python estimator
    clf_pure_predict = convert_estimator(clf)
    with open("model.pkl", "wb") as f: 
        pickle.dump(clf_pure_predict, f) 
        
    # make prediction with sklearn estimator
    y_pred = clf.predict([[0.25, 2.0, 8.3, 1.0]])
    print(y_pred)
    [2]
    
In a python enviornment with only ``pure-predict`` installed:

.. code-block:: python

    import pickle
    
    # load pickled model
    with open("model.pkl", "rb") as f: 
        clf = pickle.load(f) 
        
    # make prediction with pure-predict object
    y_pred = clf.predict([[0.25, 2.0, 8.3, 1.0]])
    print(y_pred)
    [2]

Subpackages
-----------

`pure_sklearn <https://github.com/Ibotta/pure-predict/tree/master/pure_sklearn>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Prediction in pure python for a subset of ``scikit-learn`` estimators and transformers.

- **estimators**
    - **linear models** - supports the majority of linear models for classification
    - **trees** - decision trees, random forests, gradient boosting and xgboost 
    - **naive bayes** - a number of popular naive bayes classifiers
    - **svm** - linear SVC
- **transformers**
    - **preprocessing** - normalization and onehot/ordinal encoders
    - **impute** - simple imputation 
    - **feature extraction** - text (tfidf, count vectorizer, hashing vectorizer) and dictionary vectorization 
    - **pipeline** - pipelines and feature unions

Sparse data - supports a custom pure python sparse data object - sparse data is handled as would be expected by the relevent transformers and estimators
 
`pure_fasttext <https://github.com/Ibotta/pure-predict/tree/master/pure_fasttext>`__
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Prediction in pure python for ``fasttext``.

- **supervised** - predicts labels for supervised models; no support for quantized models (blocked by `this issue <https://github.com/facebookresearch/fastText/issues/984>`__)
- **unsupervised** - lookup of word or sentence embeddings given input text

Installation
------------

Dependencies
~~~~~~~~~~~~

``pure-predict`` requires:

-  `Python <https://www.python.org/>`__ (>= 3.6)

Dependency Notes
~~~~~~~~~~~~~~~~

-  ``pure_sklearn`` has been tested with ``scikit-learn`` versions >= 0.20 -- certain functionality may work with lower versions but are not guaranteed. Some functionality is explicitly not supported for certain ``scikit-learn`` versions and exceptions will be raised as appropriate.
- ``xgboost`` requires version >= 0.82 for support with ``pure_sklearn``.
- ``pure-predict`` is not supported with Python 2.
- ``fasttext`` versions <= 0.9.1 have been tested.

User Installation
~~~~~~~~~~~~~~~~~

The easiest way to install ``pure-predict`` is with ``pip``:

::

    pip install --upgrade pure-predict

You can also download the source code:

::

    git clone https://github.com/Ibotta/pure-predict.git

Testing
~~~~~~~

With ``pytest`` installed, you can run tests locally:

::

    pytest pure-predict

Examples
--------

The package contains `examples <https://github.com/Ibotta/pure-predict/tree/master/examples>`__ 
on how to use ``pure-predict`` in practice.

Calls for Contributors
----------------------

Contributing to ``pure-predict`` is `welcomed by any contributors <https://github.com/Ibotta/pure-predict/blob/master/CONTRIBUTING.md>`__. Specific calls for contribution are as follows:

#. Examples, tests and documentation -- particularly more detailed examples with performance testing of various estimators under various constraints.
#. Adding more ``pure_sklearn`` estimators. The ``scikit-learn`` package is extensive and only partially covered by ``pure_sklearn``. `Regression <https://scikit-learn.org/stable/supervised_learning.html#supervised-learning>`__ tasks in particular missing from ``pure_sklearn``. `Clustering <https://scikit-learn.org/stable/modules/clustering.html#clustering>`__, `dimensionality reduction <https://scikit-learn.org/stable/modules/decomposition.html#decompositions>`__, `nearest neighbors <https://scikit-learn.org/stable/modules/neighbors.html>`__, `feature selection <https://scikit-learn.org/stable/modules/feature_selection.html>`__, non-linear `SVM <https://scikit-learn.org/stable/modules/svm.html>`__, and more are also omitted and would be good candidates for extending ``pure_sklearn``.
#. General efficiency. There is likely low hanging fruit for improving the efficiency of the ``numpy`` and ``scipy`` functionality that has been ported to ``pure-predict``.
#. `Threading <https://docs.python.org/3/library/threading.html>`__ could be considered to improve performance -- particularly for making predictions with multiple records.
#. A public `AWS lambda layer <https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html>`__ containing ``pure-predict``.

Background
----------

The project was started at `Ibotta
Inc. <https://medium.com/building-ibotta>`__ on the machine learning
team and open sourced in 2020. It is currently maintained by the machine 
learning team at Ibotta.

Acknowledgements
~~~~~~~~~~~~~~~~
Thanks to `David Mitchell <https://github.com/dlmitchell>`__ and `Andrew Tilley <https://github.com/tilleyand>`__ for internal review before open source. Thanks to `James Foley <https://github.com/chadfoley36>`__ for logo artwork.


.. figure:: https://github.com/Ibotta/pure-predict/blob/master/doc/images/ibottaml.png
   :alt: IbottaML

.. |License| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://opensource.org/licenses/Apache-2.0
.. |Build Status| image:: https://travis-ci.com/Ibotta/pure-predict.png?branch=master
   :target: https://travis-ci.com/Ibotta/pure-predict
.. |PyPI Package| image:: https://badge.fury.io/py/pure-predict.svg
   :target: https://pypi.org/project/pure-predict/
.. |Downloads| image:: https://pepy.tech/badge/pure-predict
   :target: https://pepy.tech/project/pure-predict
.. |Python Versions| image:: https://img.shields.io/pypi/pyversions/pure-predict
   :target: https://pypi.org/project/pure-predict/
