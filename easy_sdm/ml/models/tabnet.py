from pytorch_tabnet.tab_model import TabNetClassifier

from ..metrics.auc import AUC
from ..metrics.kappa import Kappa
from ..metrics.tss import TSS


class TabNet(TabNetClassifier):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.framework = "pytorch"
        self.estimator_name = "TabNet"

    def fit(self, X_train, y_train, X_valid, y_valid, **kwargs):

        super().fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric=[Kappa, AUC, TSS],
            **kwargs
        )

    def predict(self, x):
        return super().predict(x)

    def predict_adaptability(self, x):
        return super().predict_proba(x)[:, 1].reshape(-1, 1)
