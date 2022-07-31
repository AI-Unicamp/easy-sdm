from sklearn.metrics import cohen_kappa_score

from .base import BaseMetric


class Kappa(BaseMetric):
    def __init__(self):
        self._name = "kappa"
        self._maximize = True

    def __call__(self, y_true, y_score):
        y_score = self._adjust_y_score(y_score=y_score)
        kappa = cohen_kappa_score(y_true, (y_score >= 0.5).astype(int))
        return kappa
