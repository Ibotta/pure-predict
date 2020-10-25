"""
Performance comparison between sklearn `RandomForestClassifier`
and pure_sklearn `RandomForestClassifierPure`. In the case of
model object size, unpickle latency, and prediction latency
for a single record, we see outperformance with pure_sklearn.

For the case of trees -- pure_sklearn generally does outpeform.
Here we see a large feature space (400 features), full depth
trees and 100 estimators. The size of the pure_sklearn model 
object is half of that for sklearn -- and the prediction
latency for a single record is 1/10th that of sklearn.

We will see performance diminish as the number of records
for prediction increases.

Example Run
-----------
Pickle Size sklearn: 143769
Pickle Size pure-predict: 77603
Difference: 0.5397756122669004
Unpickle time sklearn: 0.002790212631225586
Unpickle time pure-predict: 0.0009033679962158203
Difference: 0.32376313765701104
Predict 1 record sklearn: 0.00843501091003418
Predict 1 record pure-predict: 0.0008668899536132812
Difference: 0.1027728313406258
"""

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

from pure_sklearn.utils import performance_comparison
from pure_sklearn.map import convert_estimator

N_ESTIMATORS = 100
MAX_DEPTH = None
FEATURE_MULT = 100

X, y = load_iris(return_X_y=True)
X = np.hstack([X] * FEATURE_MULT)
X_ = X.tolist()
clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH)
clf.fit(X, y)
clf_ = convert_estimator(clf)
performance_comparison(clf, clf_, X)
