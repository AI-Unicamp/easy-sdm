from configs import configs

from easy_sdm.enums import EstimatorType

from ..models import (
    MLP,
    OCSVM,
    GradientBoosting,
    RandomForest,
    TabNet,
    Xgboost,
    XgboostRF,
)


class EstimatorSelector:
    def __init__(self, estimator_type: EstimatorType) -> None:
        self.random_state = 1
        self.estimator_type = estimator_type

    def select_estimator(self):

        if self.estimator_type == EstimatorType.MLP:
            estimator = MLP(
                hidden_layer_sizes=(200, 100, 50, 20, 10),
                random_state=self.random_state,
                max_iter=8000,
            )
        elif self.estimator_type == EstimatorType.GradientBoosting:
            estimator = GradientBoosting(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state,
                criterion="squared_error",
            )
        elif self.estimator_type == EstimatorType.RandomForest:
            estimator = RandomForest(max_depth=10, random_state=self.random_state)
        elif self.estimator_type == EstimatorType.Tabnet:
            estimator = TabNet(device_name="cpu")
        elif self.estimator_type == EstimatorType.Xgboost:
            estimator = Xgboost(use_label_encoder=False)
        elif self.estimator_type == EstimatorType.XgboostRF:
            estimator = XgboostRF(use_label_encoder=False)
        elif self.estimator_type == EstimatorType.OCSVM:
            estimator = OCSVM(
                nu=configs["OCSVM"]["nu"],
                kernel=configs["OCSVM"]["kernel"],
                gamma=configs["OCSVM"]["gamma"],
            )
        else:
            raise ValueError("Use one of the possible estimators")

        self.estimator = estimator

    def get_estimator(self):
        return self.estimator

    def get_estimator_parameters(self):
        params = self.estimator.__dict__
        params = {k: v for k, v in params.items() if len(str(v)) <= 250}
        return params
