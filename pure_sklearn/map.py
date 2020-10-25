"""
Mappings from `sklearn` class names to `pure_sklearn` equivalents
"""

MAPPING = {
    "LogisticRegression": "pure_sklearn.linear_model.LogisticRegressionPure",
    "RidgeClassifier": "pure_sklearn.linear_model.RidgeClassifierPure",
    "SGDClassifier": "pure_sklearn.linear_model.SGDClassifierPure",
    "Perceptron": "pure_sklearn.linear_model.PerceptronPure",
    "PassiveAggressiveClassifier": "pure_sklearn.linear_model.PassiveAggressiveClassifierPure",
    "LinearSVC": "pure_sklearn.svm.LinearSVCPure",
    "DecisionTreeClassifier": "pure_sklearn.tree.DecisionTreeClassifierPure",
    "DecisionTreeRegressor": "pure_sklearn.tree.DecisionTreeRegressorPure",
    "ExtraTreeClassifier": "pure_sklearn.tree.ExtraTreeClassifierPure",
    "ExtraTreeRegressor": "pure_sklearn.tree.ExtraTreeRegressorPure",
    "RandomForestClassifier": "pure_sklearn.ensemble.RandomForestClassifierPure",
    "BaggingClassifier": "pure_sklearn.ensemble.BaggingClassifierPure",
    "GradientBoostingClassifier": "pure_sklearn.ensemble.GradientBoostingClassifierPure",
    "XGBClassifier": "pure_sklearn.xgboost.XGBClassifierPure",
    "ExtraTreesClassifier": "pure_sklearn.ensemble.ExtraTreesClassifierPure",
    "GaussianNB": "pure_sklearn.naive_bayes.GaussianNBPure",
    "MultinomialNB": "pure_sklearn.naive_bayes.MultinomialNBPure",
    "ComplementNB": "pure_sklearn.naive_bayes.ComplementNBPure",
    "SimpleImputer": "pure_sklearn.impute.SimpleImputerPure",
    "MissingIndicator": "pure_sklearn.impute.MissingIndicatorPure",
    "DummyClassifier": "pure_sklearn.dummy.DummyClassifierPure",
    "Pipeline": "pure_sklearn.pipeline.PipelinePure",
    "FeatureUnion": "pure_sklearn.pipeline.FeatureUnionPure",
    "OneHotEncoder": "pure_sklearn.preprocessing.OneHotEncoderPure",
    "OrdinalEncoder": "pure_sklearn.preprocessing.OrdinalEncoderPure",
    "StandardScaler": "pure_sklearn.preprocessing.StandardScalerPure",
    "MinMaxScaler": "pure_sklearn.preprocessing.MinMaxScalerPure",
    "MaxAbsScaler": "pure_sklearn.preprocessing.MaxAbsScalerPure",
    "Normalizer": "pure_sklearn.preprocessing.NormalizerPure",
    "DictVectorizer": "pure_sklearn.feature_extraction.DictVectorizerPure",
    "TfidfVectorizer": "pure_sklearn.feature_extraction.text.TfidfVectorizerPure",
    "CountVectorizer": "pure_sklearn.feature_extraction.text.CountVectorizerPure",
    "TfidfTransformer": "pure_sklearn.feature_extraction.text.TfidfTransformerPure",
    "HashingVectorizer": "pure_sklearn.feature_extraction.text.HashingVectorizerPure",
}


def _instantiate_class(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


def convert_estimator(est, min_version=None):
    """ Convert scikit-learn estimator to its pure_sklearn counterpart """
    est_name = est.__class__.__name__
    pure_est_name = MAPPING.get(est_name)
    if pure_est_name is None:
        raise ValueError(
            "Cannot find 'pure_sklearn' counterpart for {}".format(est_name)
        )
    module = ".".join(pure_est_name.split(".")[:-1])
    name = pure_est_name.split(".")[-1]
    return _instantiate_class(module, name)(est)
