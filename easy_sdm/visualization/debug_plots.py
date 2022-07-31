from pathlib import Path
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from easy_sdm.utils.data_loader import NumpyArrayLoader
from raster_processing import RasterInfoExtractor
from typos import Species

from easy_sdm.raster_processing.processing.raster_information_extractor import (
    RasterInfoExtractor,
)
from easy_sdm.utils import DatasetLoader, PathUtils, RasterLoader
from easy_sdm.utils.path_utils import PathUtils


def result_colormap():
    norm = matplotlib.colors.Normalize(-0.001, 1)
    colors = [
        [norm(-0.001), "white"],
        [norm(0.15), "0.95"],
        [norm(0.2), "sienna"],
        [norm(0.3), "wheat"],
        [norm(0.5), "cornsilk"],
        [norm(0.95), "yellowgreen"],
        [norm(1.0), "green"],
    ]

    custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    custom_cmap.set_bad(color="white")
    return custom_cmap


def ocsvm_decision_colormap():
    norm = matplotlib.colors.Normalize(-0.001, 1)
    colors = [
        [norm(-0.001), "white"],
        [norm(0.1), "0.95"],
        [norm(0.5), "red"],
        [norm(1.0), "blue"],
    ]

    custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    custom_cmap.set_bad(color="white")
    return custom_cmap


def env_var_exploration_colormap():

    norm = matplotlib.colors.Normalize(-1000, 1000)
    colors = [
        [norm(-1000), "white"],
        [norm(-100), "brown"],
        [norm(-10), "red"],
        [norm(-1), "orange"],
        [norm(0), "yellow"],
        [norm(1), "green"],
        [norm(10), "blue"],
        [norm(100), "purple"],
        [norm(1000), "black"],
    ]
    custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    custom_cmap.set_bad(color="white")
    return custom_cmap


class MapPersistance:
    def __init__(
        self, data_dirpath: Path, species: Union[Species, None], custom_cmap=None
    ) -> None:
        self.species = species
        self.data_dirpath = data_dirpath
        self.info_extractor = RasterInfoExtractor(
            RasterLoader(
                data_dirpath / "raster_processing/region_mask.tif"
            ).load_dataset()
        )
        self.__setup_vectors()
        self.custom_cmap = custom_cmap

    def __setup_vectors(self):

        xgrid = self.info_extractor.get_xgrid()
        ygrid = self.info_extractor.get_ygrid()

        X, Y = xgrid, ygrid[::-1]

        self.X = X
        self.Y = Y
        self.land_reference_array = self.info_extractor.get_array()


class MapResultsPersistanceWithoutCoords(MapPersistance):
    def __init__(
        self,
        data_dirpath: Path,
        species: Union[Species, None],
        custom_cmap=result_colormap(),
    ) -> None:
        super().__init__(data_dirpath, species, custom_cmap)

    def plot_map(
        self,
        Z: np.ndarray,
        estimator_type_text: str,
        vif_columns_identifier: str,
        experiment_dirpath: str,
    ):

        output_dirpath = (
            self.data_dirpath
            / f"visualization/map_predictions/{self.species.get_name_for_paths()}/{estimator_type_text}/{vif_columns_identifier}"
        )

        PathUtils.create_folder(output_dirpath)
        output_path = (
            output_dirpath
            / f"{self.species.get_name_for_paths()}_{estimator_type_text}_{vif_columns_identifier}_map_without_coords.png"
        )

        plt.figure(figsize=(9, 8))

        plt.ylabel("Latitude[degrees]", fontsize=26)
        plt.xlabel("Longitude[degrees]", fontsize=26)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        # Plot country map
        plt.contour(
            self.X,
            self.Y,
            self.land_reference_array,
            levels=[10],
            colors="k",
            linestyles="solid",
        )

        plt.contourf(self.X, self.Y, Z, levels=10, cmap=self.custom_cmap)

        clb = plt.colorbar(format="%.2f")
        clb.ax.tick_params(labelsize=20)

        # Saving results
        plt.legend(loc="upper right")

        PathUtils.create_folder(output_dirpath)
        plt.savefig(output_path)
        plt.clf()
        return output_path


class MapResultsPersistanceWithCoords(MapPersistance):
    def __init__(
        self,
        data_dirpath: Path,
        species: Union[Species, None],
        custom_cmap=result_colormap(),
    ) -> None:
        super().__init__(data_dirpath, species, custom_cmap)

    def _extract_occurrence_coords(self, experiment_dirpath):
        coords_df, _ = DatasetLoader(
            experiment_dirpath / "coords_df.csv"
        ).load_dataset()
        sdm_df, _ = DatasetLoader(experiment_dirpath / "complete_df.csv").load_dataset()
        df_occ = sdm_df.loc[sdm_df["label"] == 1]
        coords_occ_df = coords_df.iloc[list(df_occ.index)]
        coords = coords_occ_df.to_numpy()
        return coords

    def _extract_pseudo_absense_coords(self, experiment_dirpath):
        coords_df, _ = DatasetLoader(
            experiment_dirpath / "coords_df.csv"
        ).load_dataset()
        sdm_df, _ = DatasetLoader(experiment_dirpath / "complete_df.csv").load_dataset()
        df_psa = sdm_df.loc[sdm_df["label"] == 0]
        coords_psa_df = coords_df.iloc[list(df_psa.index)]
        coords = coords_psa_df.to_numpy()
        return coords

    def plot_map(
        self,
        Z: np.ndarray,
        estimator_type_text: str,
        vif_columns_identifier: str,
        experiment_dirpath: str,
    ):

        output_dirpath = (
            self.data_dirpath
            / f"visualization/map_predictions/{self.species.get_name_for_paths()}/{estimator_type_text}/{vif_columns_identifier}"
        )

        PathUtils.create_folder(output_dirpath)
        output_path = (
            output_dirpath
            / f"{self.species.get_name_for_paths()}_{estimator_type_text}_{vif_columns_identifier}_map_with_coords.png"
        )

        plt.figure(figsize=(9, 8))

        plt.ylabel("Latitude[degrees]", fontsize=26)
        plt.xlabel("Longitude[degrees]", fontsize=26)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        # Plot country map
        plt.contour(
            self.X,
            self.Y,
            self.land_reference_array,
            levels=[10],
            colors="k",
            linestyles="solid",
        )

        # print('levels: ',levels)
        plt.contourf(self.X, self.Y, Z, levels=10, cmap=self.custom_cmap)
        plt.colorbar(format="%.2f")

        occ_coords = self._extract_occurrence_coords(experiment_dirpath)
        psa_coords = self._extract_pseudo_absense_coords(experiment_dirpath)

        plt.scatter(
            occ_coords[:, 1],
            occ_coords[:, 0],
            s=2 ** 3,
            c="blue",
            marker="^",
            label="Occurrance coordinates",
        )
        plt.scatter(
            psa_coords[:, 1],
            psa_coords[:, 0],
            s=2 ** 3,
            c="red",
            marker="^",
            label="Pseudo absenses coordinates",
        )

        # Saving results
        plt.legend(loc="upper right")
        plt.savefig(output_path)
        plt.clf()
        return output_path


class MapWithCoords(MapPersistance):
    def __init__(
        self, data_dirpath: Path, species: Union[Species, None], custom_cmap=None
    ) -> None:
        super().__init__(data_dirpath, species, custom_cmap)
        self.experiment_dirpath = (
            self.data_dirpath
            / f"featuarizer/datasets/{self.species.get_name_for_paths()}/binary_classification/rsep/full_data"
        )

    def _extract_occurrence_coords(self):
        coords_df, _ = DatasetLoader(
            self.experiment_dirpath / "coords_df.csv"
        ).load_dataset()

        sdm_df, _ = DatasetLoader(
            self.experiment_dirpath / "complete_df.csv"
        ).load_dataset()

        df_occ = sdm_df.loc[sdm_df["label"] == 1]
        coords_occ_df = coords_df.iloc[list(df_occ.index)]
        coords = coords_occ_df.to_numpy()
        return coords

    def _extract_pseudo_absense_coords(self):
        coords_df, _ = DatasetLoader(
            self.experiment_dirpath / "coords_df.csv"
        ).load_dataset()
        sdm_df, _ = DatasetLoader(
            self.experiment_dirpath / "complete_df.csv"
        ).load_dataset()
        df_psa = sdm_df.loc[sdm_df["label"] == 0]
        coords_psa_df = coords_df.iloc[list(df_psa.index)]
        coords = coords_psa_df.to_numpy()
        return coords

    def plot_map(self,):

        output_dirpath = (
            self.data_dirpath / f"eda/species/{self.species.get_name_for_paths()}"
        )

        PathUtils.create_folder(output_dirpath)
        output_path = (
            output_dirpath / f"{self.species.get_name_for_paths()}_cords_in_map.png"
        )

        plt.figure(figsize=(9, 8))

        plt.ylabel("Latitude[degrees]", fontsize=26)
        plt.xlabel("Longitude[degrees]", fontsize=26)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        # Plot country map
        plt.contour(
            self.X,
            self.Y,
            self.land_reference_array,
            levels=[10],
            colors="k",
            linestyles="solid",
        )

        occ_coords = self._extract_occurrence_coords()
        psa_coords = self._extract_pseudo_absense_coords()

        plt.scatter(
            occ_coords[:, 1],
            occ_coords[:, 0],
            s=2 ** 3,
            c="blue",
            marker="^",
            label="OCC coordinates",
        )
        plt.scatter(
            psa_coords[:, 1],
            psa_coords[:, 0],
            s=2 ** 3,
            c="red",
            marker="^",
            label="PSA coordinates",
        )

        # Saving results
        plt.legend(loc="lower left", prop={"size": 17})
        plt.savefig(output_path)
        plt.clf()


class MapWithOCSVMDecision(MapPersistance):
    def __init__(
        self,
        data_dirpath: Path,
        species: Union[Species, None],
        custom_cmap=ocsvm_decision_colormap(),
    ) -> None:
        super().__init__(data_dirpath, species, custom_cmap)
        self.experiment_dirpath = (
            self.data_dirpath
            / f"featuarizer/datasets/{self.species.get_name_for_paths()}/binary_classification/rsep/full_data"
        )

    def plot_map(self,):

        output_dirpath = (
            self.data_dirpath / f"eda/species/{self.species.get_name_for_paths()}"
        )

        PathUtils.create_folder(output_dirpath)
        output_path = (
            output_dirpath
            / f"{self.species.get_name_for_paths()}_ocsvm_decision_map.png"
        )

        plt.figure(figsize=(9, 8))

        plt.ylabel("Latitude[degrees]", fontsize=26)
        plt.xlabel("Longitude[degrees]", fontsize=26)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        Z = NumpyArrayLoader(
            self.experiment_dirpath / "psa_decision_map.npy"
        ).load_dataset()
        Z[Z == -9999.0] = -0.001
        Z[Z == -1] = 0.5

        plt.contourf(self.X, self.Y, Z, levels=10, cmap=self.custom_cmap)
        # Plot country map
        plt.contour(
            self.X,
            self.Y,
            self.land_reference_array,
            levels=[10],
            colors="k",
            linestyles="solid",
        )

        # Saving results
        plt.savefig(output_path)
        plt.clf()
        return output_path


class EnvironmentVariablesMapPlotter(MapPersistance):
    def __init__(
        self,
        data_dirpath: Path,
        species=None,
        custom_cmap=env_var_exploration_colormap(),
    ) -> None:
        super().__init__(data_dirpath, None, custom_cmap)

    def plot_map(self, Z: np.ndarray, variable_name: str):
        print(Z.min())
        print(Z.max())

        Z[Z == -9999.0] = -1000
        Z[0, 0] = 1000

        plt.figure(figsize=(9, 8))

        plt.ylabel("Latitude[degrees]", fontsize=26)
        plt.xlabel("Longitude[degrees]", fontsize=26)

        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        # Plot country map
        plt.contour(
            self.X,
            self.Y,
            self.land_reference_array,
            levels=[10],
            colors="k",
            linestyles="solid",
        )

        # print('levels: ',levels)
        plt.contourf(
            self.X, self.Y, Z, levels=10, cmap=self.custom_cmap, vmin=-1000, vmax=1000
        )
        plt.colorbar(format="%.2f")

        # Saving results
        plt.legend(loc="upper right")
        output_dirpath = self.data_dirpath / "eda" / "env_variables_plots"
        PathUtils.create_folder(output_dirpath)
        output_path = output_dirpath / variable_name
        plt.savefig(output_path)
        plt.clf()
