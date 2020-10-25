"""
Performance comparison between sklearn `LogisticRegression`
and pure_sklearn `LogisticRegressionPure`. In the case of
model object size, unpickle latency, and prediction latency
for a single record, we see outperformance with pure_sklearn.

This is meant to illustrate that the pure python implementation
can outperform sklearn. Note that this will not always be the case,
especially as the number of features and classes increases and as
the number of records to predict increases.

Example Run
-----------
Pickle Size sklearn: 957
Pickle Size pure-predict: 343
Difference: 0.35841170323928945
Unpickle time sklearn: 8.702278137207031e-05
Unpickle time pure-predict: 4.696846008300781e-05
Difference: 0.5397260273972603
Predict 1 record sklearn: 0.00021004676818847656
Predict 1 record pure-predict: 7.390975952148438e-05
Difference: 0.3518728717366629
"""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

from pure_sklearn.utils import performance_comparison
from pure_sklearn.map import convert_estimator

SOLVER = "lbfgs"
MAX_ITER = 1000
MULTI_CLASS = "ovr"
METHOD = "predict_proba"
FEATURE_MULT = 1

X, y = load_iris(return_X_y=True)
X = np.hstack([X] * FEATURE_MULT)
X_ = X.tolist()
clf = LogisticRegression(solver=SOLVER, multi_class=MULTI_CLASS, max_iter=MAX_ITER)
clf.fit(X, y)
clf_ = convert_estimator(clf)
performance_comparison(clf, clf_, X)
