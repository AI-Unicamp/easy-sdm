from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from easy_sdm.utils.data_loader import DatasetLoader, PickleLoader


class VifRelevantInfoSelector:
    def __init__(self, data_dirpath) -> None:
        self.data_dirpath = data_dirpath
        self.__setup_mlflow()
        self.experiment_dataset_path = None
        self.vif_decision_columns = None

    def __setup_mlflow(self):
        ml_dirpath = str(Path.cwd() / "data/ml")
        mlflow.set_tracking_uri(f"file:{ml_dirpath}")

    def build_vif_info_from_experiment_path(self, path: Path):
        self.experiment_dataset_path = path
        self.__build_vif()

    def build_vif_info_from_runid(self, run_id: int):
        self.experiment_dataset_path = Path(
            mlflow.get_run(run_id).data.tags["experiment_featurizer_path"]
        )
        self.__build_vif()

    def __build_vif(self):
        relevant_raster_list = PickleLoader(
            self.data_dirpath / "environment/relevant_raster_list"
        ).load_dataset()

        vif_decision_df, _ = DatasetLoader(
            self.experiment_dataset_path / "vif_decision_df.csv"
        ).load_dataset()
        self.vif_decision_columns = vif_decision_df["feature"].tolist()
        relevant_raster_name_list = [
            str(path).split("/")[-1].replace(".tif", "")
            for path in relevant_raster_list
        ]
        self.vif_relevant_raster_list_pos = [
            relevant_raster_name_list.index(elem) for elem in self.vif_decision_columns
        ]

    def filter_vif_from_stack(self, stack: np.array):
        filtered_stack = stack[self.vif_relevant_raster_list_pos]
        return filtered_stack

    def filter_vif_from_statistics(self, statistics_dataset: pd.DataFrame):
        filtered_statistics_dataset = statistics_dataset[
            statistics_dataset["raster_name"].isin(self.vif_decision_columns)
        ]
        return filtered_statistics_dataset

    def filter_vif_from_dataset_features(self, X: pd.DataFrame):
        return X.iloc[:, self.vif_relevant_raster_list_pos]
