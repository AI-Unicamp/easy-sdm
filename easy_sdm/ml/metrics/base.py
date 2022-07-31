from pytorch_tabnet.metrics import Metric


class BaseMetric(Metric):
    def _adjust_y_score(self, y_score):
        """Get only the positive probabilities

        Args:
            y_score (numpy_array): Can be 1D or 2D.  Must be the predict proba of positive cases.
        """
        # tabnet inheritance situation
        if y_score.shape[1] == 2:
            y_score = y_score[:, 1]
        # getmetics situation
        elif y_score.shape[1] == 1:
            # y_score = y_score.ravel()
            pass
        else:
            ValueError("y_score shape cant be zero")

        return y_score

    def __call__(self, y_true, y_score):
        NotImplementedError()
