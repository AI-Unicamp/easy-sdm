from sklearn.svm import OneClassSVM

# nu: percent of anomalies


class OCSVM(OneClassSVM):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.framework = "sklearn"
        self.estimator_name = "OCSVM"

    def fit(self, X_train, y_train=None, X_valid=None, y_valid=None, **kwargs):
        super().fit(X_train, **kwargs)

    def predict(self, x):
        return super().predict(x)

    def predict_adaptability(self, x):
        return super().decision_function(x).reshape(-1, 1)
