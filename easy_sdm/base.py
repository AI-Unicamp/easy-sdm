from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd
from sklearn.base import BaseEstimator


class BaseEstimatorPersistence(ABC):
    @classmethod
    @abstractmethod
    def dump(
        cls,
        estimator: BaseEstimator,
        output_dir: str,
        filename: str = "model",
        suffix: str = "pkl",
        **kwargs,
    ) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, file_path: str, **kwargs) -> BaseEstimator:
        pass


class BaseExperimentPersistence(ABC):
    @classmethod
    @abstractmethod
    def save_scores(
        cls, metrics: Dict[str, float], output_dir: str, file_name: str
    ) -> None:
        pass

    @classmethod
    @abstractmethod
    def save_predictions(
        cls,
        predictions: pd.DataFrame,
        output_dir: str,
        file_name: str,
        suffix: str = "csv",
    ) -> None:
        pass


class BaseDataPersistence(ABC):
    @classmethod
    @abstractmethod
    def training_dataset_profiling(
        cls, value: pd.DataFrame, output_dir: str, **kwargs
    ) -> pd.DataFrame:
        pass

    @classmethod
    @abstractmethod
    def features_payload_profiling(
        cls, value: pd.DataFrame, output_dir: str, **kwargs,
    ) -> None:
        pass


class BaseDataLoader(ABC):
    @abstractmethod
    def load_dataset(data_path: str):
        pass


class BaseTrainJob(ABC):
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def persist(self):
        pass
