import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typos import Species

from easy_sdm.enums import EstimatorType
from easy_sdm.raster_processing.processing.raster_information_extractor import (
    RasterInfoExtractor,
)
from easy_sdm.utils.path_utils import PathUtils


class MapPlotter(object):
    def __init__(self, species: Species, data_dirpath: Path) -> None:
        self.species = species
        self.data_dirpath = data_dirpath
        self.info_extractor = RasterInfoExtractor(
            data_dirpath / "raster_processing/region_mask.tif"
        )
        self.__setup_vectors()
        self.__setup_colormap()

    def __setup_vectors(self):

        xgrid = self.info_extractor.get_xgrid()
        ygrid = self.info_extractor.get_ygrid()

        X, Y = xgrid, ygrid[::-1]

        self.X = X
        self.Y = Y
        self.land_reference_array = self.info_extractor.get_array()

    def __setup_colormap(self):
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
        self.custom_cmap = custom_cmap

    def create_result_adaptabilities_map(
        self, Z: np.ndarray, estimator_type: EstimatorType, run_id: str,
    ):

        plt.figure(figsize=(8, 8))

        # Setting titles and labels
        plt.title(
            f"Distribuição predita para a \nespécie {self.species.get_name_for_plots()}\n algoritimo {estimator_type.name}",
            fontsize=20,
        )
        plt.ylabel("Latitude[graus]", fontsize=18)
        plt.xlabel("Longitude[graus]", fontsize=18)

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

        # Saving results
        plt.legend(loc="upper right")
        output_dirpath = (
            self.data_dirpath / f"output/{self.species.get_name_for_paths()}"
        )
        PathUtils.create_folder(output_dirpath)
        output_path = output_dirpath / f"{estimator_type.name}_{run_id}"
        plt.savefig(output_path)
        plt.show()
        plt.clf()

    def create_result_adaptabilities_map_wtih_coords(self):
        raise NotImplementedError()
