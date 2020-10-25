"""
Performance comparison between sklearn and pure_sklearn for a 
text pipeline. The pipeline unions a `TfidfVectorizer` and a `HashingVectorizer`
followed by a `RandomForestClassifier` as the estimator.
In the case of model object size, unpickle latency, 
and prediction latency for a single record, we see 
outperformance with pure_sklearn.

We see substantial outperformance with pure_sklearn for
single record prediction.

Example Run
-----------
Pickle Size sklearn: 5546533
Pickle Size pure-predict: 3334779
Difference: 0.6012366644172135
Unpickle time sklearn: 0.048230886459350586
Unpickle time pure-predict: 0.03762626647949219
Difference: 0.7801280308460417
Predict 1 record sklearn: 0.09419584274291992
Predict 1 record pure-predict: 0.00426483154296875
Difference: 0.04527621834233559
"""

import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_20newsgroups

from pure_sklearn.utils import performance_comparison
from pure_sklearn.map import convert_estimator

N_ESTIMATORS = 100
MAX_DEPTH = None

categories = ["rec.autos", "sci.space"]
X, y = fetch_20newsgroups(subset="train", categories=categories, return_X_y=True)
vec1 = HashingVectorizer()
vec2 = TfidfVectorizer()
feats = FeatureUnion([("vec1", vec1), ("vec2", vec2)])
rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH)
clf = Pipeline(steps=[("feats", feats), ("rf", rf)])
clf.fit(X, y)
clf_ = convert_estimator(clf)
performance_comparison(clf, clf_, X)
