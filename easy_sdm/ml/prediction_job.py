from pathlib import Path

import mlflow
import numpy as np

from easy_sdm.configs import configs
from easy_sdm.featuarizer import MinMaxScalerWrapper
from easy_sdm.typos.species import Species
from easy_sdm.utils.data_loader import DatasetLoader, NumpyArrayLoader, RasterLoader
from easy_sdm.visualization.debug_plots import (
    MapResultsPersistanceWithCoords,
    MapResultsPersistanceWithoutCoords,
)

from .selectors.vif_relevant_info_selector import VifRelevantInfoSelector


class Prediction_Job:

    """
    the output directory is prepared before logging:
    ├── output
    │   ├── data
    │   │   ├── data_sample.csv
    │   │   └── data_sample.html
    │   ├── images
    │   │   ├── gif_sample.gif
    │   │   └── image_sample.png
    │   ├── maps
    │   │   └── map_sample.geojson
    │   └── plots
    │       └── plot_sample.html
    """

    def __init__(self, data_dirpath: Path, run_id: str, species: Species) -> None:
        self.configs = configs
        self.data_dirpath = data_dirpath
        self.species = species
        self.run_id = run_id
        self.__setup_mlflow()
        self.__setup()

    def __setup(self):

        self.tags = mlflow.get_run(self.run_id).data.tags

        land_reference = RasterLoader(
            self.data_dirpath / "raster_processing/region_mask.tif"
        ).load_dataset()

        statistics_dataset, _ = DatasetLoader(
            self.data_dirpath / "environment/raster_statistics.csv"
        ).load_dataset()

        if self.tags["vif"] == "vif_columns":
            self.vif_relevant_info_selector = VifRelevantInfoSelector(
                data_dirpath=self.data_dirpath
            )
            self.vif_relevant_info_selector.build_vif_info_from_runid(
                run_id=self.run_id
            )
            statistics_dataset = self.vif_relevant_info_selector.filter_vif_from_statistics(
                statistics_dataset=statistics_dataset
            )

        self.scaler = MinMaxScalerWrapper(statistics_dataset=statistics_dataset)

        self.idx = np.where(
            land_reference.read(1) == self.configs["mask"]["positive_mask_val"]
        )

        self.history = (
            mlflow.get_run(self.run_id).data.tags["mlflow.log-model.history"].lower()
        )

        self.loaded_model = None

    def __setup_mlflow(self):
        ml_dirpath = str(Path.cwd() / "data/ml")
        mlflow.set_tracking_uri(f"file:{ml_dirpath}")

    def set_model(self):

        # mlflow.get_run()
        logged_model = f"runs:/{self.run_id}/{self.species.get_name_for_plots()}"
        # self.loaded_model = mlflow.pyfunc.load_model(logged_model)

        if "xgboost" in self.history:
            self.loaded_model = mlflow.xgboost.load_model(logged_model)
        elif "tabnet" in self.history:
            self.loaded_model = mlflow.pytorch.load_model(logged_model)
        else:
            self.loaded_model = mlflow.sklearn.load_model(logged_model)

    def __get_stacked_raster_coverages(self):

        stacked_raster_coverages = NumpyArrayLoader(
            self.data_dirpath / "environment/environment_stack.npy"
        ).load_dataset()

        if self.tags["vif"] == "vif_columns":
            stacked_raster_coverages = self.vif_relevant_info_selector.filter_vif_from_stack(
                stack=stacked_raster_coverages
            )

        return stacked_raster_coverages

    def map_prediction(self):

        assert self.loaded_model != None, "Set model first"

        stacked_raster_coverages = self.__get_stacked_raster_coverages()
        coverages_of_interest = stacked_raster_coverages[:, self.idx[0], self.idx[1]].T
        scaled_coverages = self.scaler.scale_coverages(coverages_of_interest)
        global_pred = self.loaded_model.predict_adaptability(scaled_coverages)

        # num_env_vars = stacked_raster_coverages.shape[0]
        # global_pred = self.loaded_model.predict(stacked_raster_coverages.reshape(num_env_vars,-1).T)

        Z = np.ones(
            (stacked_raster_coverages.shape[1], stacked_raster_coverages.shape[2]),
            dtype=np.float32,
        )
        # Z *= global_pred.min()
        # Z *=-1 #This will be necessary to set points outside map to the minimum
        Z *= self.configs["maps"][
            "no_data_val"
        ]  # This will be necessary to set points outside map to the minimum

        Z[self.idx[0], self.idx[1]] = global_pred.ravel()

        Z[Z == self.configs["maps"]["no_data_val"]] = -0.001
        return Z

    def log_map_without_coords(self, Z: np.ndarray):
        map_persistance = MapResultsPersistanceWithoutCoords(
            species=self.species, data_dirpath=self.data_dirpath
        )

        output_path = map_persistance.plot_map(
            Z=Z,
            estimator_type_text=self.tags["Estimator"],
            vif_columns_identifier=self.tags["vif"],
            experiment_dirpath=Path(self.tags["experiment_featurizer_path"]),
        )

        self.__mlflow_log_artifact(output_path=output_path)

    def log_map_with_coords(self, Z: np.ndarray):
        map_persistance = MapResultsPersistanceWithCoords(
            species=self.species, data_dirpath=self.data_dirpath
        )

        output_path = map_persistance.plot_map(
            Z=Z,
            estimator_type_text=self.tags["Estimator"],
            vif_columns_identifier=self.tags["vif"],
            experiment_dirpath=Path(self.tags["experiment_featurizer_path"]),
        )

        self.__mlflow_log_artifact(output_path=output_path)

    def __mlflow_log_artifact(self, output_path):
        with mlflow.start_run(run_id=self.run_id) as run:
            mlflow.log_artifact(output_path)

    def dataset_prediction(self):
        pass
