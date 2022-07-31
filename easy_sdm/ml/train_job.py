from pathlib import Path, PurePath
from typing import Dict

import numpy as np
import pandas as pd

from easy_sdm.enums import EstimatorType
from easy_sdm.enums.modelling_type import ModellingType
from easy_sdm.enums.pseudo_species_generators import PseudoSpeciesGeneratorType
from easy_sdm.typos import Species
from easy_sdm.utils import DatasetLoader
from easy_sdm.utils.path_utils import PathUtils

from .metrics import MetricsTracker
from .persistance.mlflow_persisance import MLFlowPersistence
from .selectors.estimator_selector import EstimatorSelector
from .selectors.vif_relevant_info_selector import VifRelevantInfoSelector


class KfoldTrainJob:
    def __init__(
        self,
        data_dirpath: Path,
        estimator_type: EstimatorType,
        modelling_type: ModellingType,
        ps_generator_type: PseudoSpeciesGeneratorType,
        species: Species,
    ) -> None:
        self.data_dirpath = data_dirpath

        self.estimator_type = estimator_type
        self.modelling_type = modelling_type
        self.ps_generator_type = ps_generator_type
        self.species = species

        self.experiment_dataset_path = data_dirpath / species.get_name_for_paths()

    def __select_etimator(self):
        estimator_selector = EstimatorSelector(self.estimator_type)
        estimator_selector.select_estimator()
        estimator = estimator_selector.get_estimator()
        estimator_parameters = estimator_selector.get_estimator_parameters()

        return estimator, estimator_parameters

    def __setup(self, only_vif_columns: bool = False):

        self.columns_considered = "vif_columns" if only_vif_columns else "all_columns"

        # SETUP ESTIMATOR
        self.estimator, self.estimator_parameters = self.__select_etimator()

        # SETTING METRICS TRACKER
        self.metrics_tracker = MetricsTracker()

        # SETTING VifRelevantInfoSelector
        self.vif_relevant_info_selector = VifRelevantInfoSelector(self.data_dirpath)

    def __setup_kfold(self, path: Path, only_vif_columns: bool = False):
        self.__setup(only_vif_columns=only_vif_columns)

        # LOAD DATASETS
        vif_prefix = "vif_" if only_vif_columns else ""
        (X_train_df, y_train_df,) = DatasetLoader(
            path / f"{vif_prefix}train.csv", output_column="label"
        ).load_dataset()
        (X_test_df, y_test_df,) = DatasetLoader(
            path / f"{vif_prefix}test.csv", output_column="label"
        ).load_dataset()

        return X_train_df, y_train_df, X_test_df, y_test_df

    def __setup_full_data(self, path: Path, only_vif_columns: bool = False):
        self.__setup(only_vif_columns=only_vif_columns)
        full_data_dataloader = DatasetLoader(
            path / "complete_df.csv", output_column="label"
        )
        (complete_X_train_df, y_train_df) = full_data_dataloader.load_dataset()

        if only_vif_columns:
            self.vif_relevant_info_selector.build_vif_info_from_experiment_path(
                path=path
            )
            (
                X_train_df
            ) = self.vif_relevant_info_selector.filter_vif_from_dataset_features(
                X=complete_X_train_df
            )
        else:
            X_train_df = complete_X_train_df

        # SETUP MLFLOW
        self.experiment_featurizer_path = full_data_dataloader.dataset_path.parent

        mlflow_experiment_name = self.species.get_name_for_plots()
        self.mlflow_persister = MLFlowPersistence(mlflow_experiment_name)

        return X_train_df, y_train_df

    def __fit(self, X_train_df: pd.DataFrame, y_train_df: pd.DataFrame):

        self.estimator.fit(X_train_df, y_train_df)

    def __validate(self, X_test_df: pd.DataFrame, y_test_df: pd.DataFrame):

        prediction_scores = self.estimator.predict_adaptability(x=X_test_df)

        metrics = self.metrics_tracker.get_metrics(
            y_true=y_test_df, y_score=prediction_scores
        )

        return metrics

    def __dataset_dirpath(self):
        generetor_type = (
            self.ps_generator_type.value if self.ps_generator_type != None else ""
        )
        dataset_dirpath = (
            self.data_dirpath
            / f"featuarizer/datasets"
            / self.species.get_name_for_paths()
            / self.modelling_type.value
            / generetor_type
        )
        return dataset_dirpath

    def __persist(self, model, metrics_statistics, parameters, kfold_metrics, tags):

        self.mlflow_persister.persist(
            model=model,
            metrics=metrics_statistics,
            parameters=parameters,
            tags=tags,
            kfold_metrics=kfold_metrics,
        )

    def __check_if_kfold_path(self, path: Path):
        result = False
        if path.is_dir() and str(path).find("kfold") != -1:
            result = True
        return result

    def __kfold_experiment(self, only_vif_columns: bool) -> Dict:

        kfold_metrics = {}
        dataset_dirpath = self.__dataset_dirpath()
        for path in Path(dataset_dirpath).iterdir():
            if self.__check_if_kfold_path(path):
                kfold_key = PurePath(path).name
                X_train_df, y_train_df, X_test_df, y_test_df = self.__setup_kfold(
                    path=path, only_vif_columns=only_vif_columns
                )
                self.__fit(X_train_df, y_train_df)
                metrics = self.__validate(X_test_df, y_test_df)
                kfold_metrics[kfold_key] = metrics
                self.__reset()

        return kfold_metrics

    def __full_data_experiment(self, only_vif_columns: bool) -> Dict:
        path = self.__dataset_dirpath() / "full_data"
        X_train_df, y_train_df = self.__setup_full_data(
            path=path, only_vif_columns=only_vif_columns
        )
        self.__fit(X_train_df, y_train_df)

    def __calculate_metrics_statistics(self, kfold_metrics: Dict):
        metrics_statistics = {}
        auc_list = []
        kappa_list = []
        tss_list = []
        for _, metrics in kfold_metrics.items():
            auc_list.append(metrics["auc"])
            kappa_list.append(metrics["kappa"])
            tss_list.append(metrics["tss"])

        metrics_statistics["auc_mean"] = np.mean(auc_list)
        metrics_statistics["kappa_mean"] = np.mean(kappa_list)
        metrics_statistics["tss_mean"] = np.mean(tss_list)

        metrics_statistics["auc_std"] = np.std(auc_list)
        metrics_statistics["kappa_std"] = np.std(kappa_list)
        metrics_statistics["tss_std"] = np.std(tss_list)

        return metrics_statistics

    def run_experiment(self, only_vif_columns: bool):
        kfold_metrics = self.__kfold_experiment(only_vif_columns)
        metrics_statistics = self.__calculate_metrics_statistics(kfold_metrics)
        self.__full_data_experiment(only_vif_columns)

        tags = {
            "vif": self.columns_considered,
            "experiment_featurizer_path": self.experiment_featurizer_path,
            "ps_generator_type": self.ps_generator_type.name,
        }

        self.__persist(
            model=self.estimator,
            parameters=self.estimator_parameters,
            metrics_statistics=metrics_statistics,
            tags=tags,
            kfold_metrics=kfold_metrics,
        )

    def __reset(self):
        self.columns_considered = None
        self.estimator = None
        self.estimator_parameters = None
        self.metrics_tracker = None
        self.mlflow_persister = None


class SimpleTrainJob:
    def __init__(
        self,
        train_data_loader: DatasetLoader,
        validation_data_loader: DatasetLoader,
        estimator_type: EstimatorType,
        species: Species,
        output_path: Path = None,
    ) -> None:
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.estimator_type = estimator_type
        self.species = species
        self.output_path = output_path

        estimator_selector = EstimatorSelector(self.estimator_type)
        estimator_selector.select_estimator()
        self.estimator = estimator_selector.get_estimator()
        self.estimator_parameters = estimator_selector.get_estimator_parameters()
        self.vif_columns = False
        self.__setup()

    def __build_empty_folders(self):
        raise NotImplementedError()

    def normal_setup(
        self, train_data_loader: DatasetLoader, validation_data_loader: DatasetLoader
    ):
        self.train_data_loader = train_data_loader
        self.validation_data_loader = validation_data_loader
        self.__setup()

    def vif_setup(
        self,
        vif_train_data_loader: DatasetLoader,
        vif_validation_data_loader: DatasetLoader,
    ):
        self.vif_columns = True
        self.train_data_loader = vif_train_data_loader
        self.validation_data_loader = vif_validation_data_loader
        self.__setup()

    def __setup(self):

        # Path with data used in the experiment
        experiment_featurizer_path = "/".join(
            str(self.train_data_loader.dataset_path).split("/")[:-1]
        )

        # SETUP MLFLOW
        self.mlflow_experiment_name = self.species.get_name_for_plots()
        self.columns_considered = "vif_columns" if self.vif_columns else "all_columns"

        self.mlflow_persister = MLFlowPersistence(
            self.mlflow_experiment_name, experiment_featurizer_path
        )

        # LOAD DATASETS
        (self.X_train_df, self.y_train_df,) = self.train_data_loader.load_dataset()
        (self.x_valid_df, self.y_valid_df,) = self.validation_data_loader.load_dataset()

        # SETTING PIPELINE
        self.pipeline = self.estimator

        # SETTING METRICS TRACKER
        self.metrics_tracker = MetricsTracker()

    def fit(self):
        self.pipeline.fit(
            self.X_train_df, self.y_train_df, self.x_valid_df, self.y_valid_df
        )
        self.__validate()

    def __validate(self):
        self.prediction_scores = self.pipeline.predict_adaptability(x=self.x_valid_df)
        self.metrics = self.metrics_tracker.get_metrics(
            y_true=self.y_valid_df, y_score=self.prediction_scores
        )

    def persist(self):

        self.mlflow_persister.persist(
            model=self.pipeline,
            metrics=self.metrics,
            parameters=self.estimator_parameters,
            vif=self.columns_considered,
        )
