from enum import Enum


class ModellingType(Enum):
    BinaryClassification = "binary_classification"  # Tabnet, MLP, EnsembleForest
    AnomalyDetection = "anomaly_detection"  # OCSVM , Autoencoder
