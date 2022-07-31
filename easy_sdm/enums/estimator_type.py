from enum import Enum


class EstimatorType(Enum):
    Tabnet = "Tabnet"
    MLP = "MLP"
    GradientBoosting = "GradientBoosting"
    RandomForest = "RandomForest"
    Xgboost = "Xgboost"
    XgboostRF = "XgboostRF"
    OCSVM = "OCSVM"
    Autoencoder = "Autoencoder"
