class BaseEstimator:
    def __init__(self, **kwargs) -> None:
        self.framework = "base"
        self.estimator_name = "base"
        raise NotImplementedError()

    def fit(self, X_train, y_train, X_valid=None, y_valid=None, **kwargs):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def predict_adaptability(self, x):
        raise NotImplementedError()
