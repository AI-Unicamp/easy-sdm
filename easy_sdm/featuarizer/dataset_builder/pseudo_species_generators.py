from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
import rasterio

from easy_sdm.configs import configs
from easy_sdm.ml import OCSVM
from pathlib import Path


class BasePseudoSpeciesGenerator(ABC):
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def fit(self, occurrence_data: pd.DataFrame):
        raise NotImplementedError()

    @abstractmethod
    def generate(self, number_pseudo_absenses: int):
        raise NotImplementedError()

    def _find_coordinate_in_map_matrix(self, matrix_position: Tuple):
        lat_coord = (
            configs["maps"]["region_limits_with_security"]["north"]
            - matrix_position[0] * configs["maps"]["resolution"]
        )
        long_coord = (
            configs["maps"]["region_limits_with_security"]["west"]
            + matrix_position[1] * configs["maps"]["resolution"]
        )
        return (lat_coord, long_coord)


class RandomPseudoSpeciesGenerator(BasePseudoSpeciesGenerator):
    def __init__(self) -> None:
        raise NotImplementedError("This class is not yet implemented")

    def generate(self):
        pass


class RSEPPseudoSpeciesGenerator(BasePseudoSpeciesGenerator):
    """[Generate points that are inside the proposed territory and in regions where data is an an anomaly]

    Args:
        BasePseudoSpeciesGenerator ([type]): [description]
    """

    def __init__(self, **kwargs) -> None:

        self.region_mask_raster = kwargs.get("region_mask_raster", None)
        self.stacked_raster_coverages = kwargs.get("stacked_raster_coverages", None)
        self.min_max_scaler = kwargs.get("min_max_scaler", None)

        self.__check_arguments()
        self.configs = configs
        self.inside_mask_idx = np.where(
            self.region_mask_raster.read(1) != configs["mask"]["negative_mask_val"]
        )
        self.ocsvm_configs = configs["OCSVM"]
        self.ocsvm = OCSVM(**self.ocsvm_configs)

    def __check_arguments(self):
        assert type(self.region_mask_raster) is rasterio.io.DatasetReader
        assert type(self.stacked_raster_coverages) is np.ndarray

    def fit(self, scaled_occurrence_df: pd.DataFrame):
        # Coords X and Y in two tuples where condition matchs (array(),array())

        scaled_occurrence_df = scaled_occurrence_df.drop("label", axis=1)
        self.ocsvm.fit(X_train=scaled_occurrence_df.values)
        self.columns = scaled_occurrence_df.columns
        self.__create_decition_points()

    def __create_decition_points(self):
        """[

            Z will be a 2D array with 3 possible values:
                # Outside Brazilian mask: -9999
                # Not valid predictions: -1 (Useful ones)
                # Valid predictions: 1
        ]

        Returns:
            [type]: [description]
        """
        Z = np.ones(
            (
                self.stacked_raster_coverages.shape[1],
                self.stacked_raster_coverages.shape[2],
            ),
            dtype=np.float32,
        )
        Z *= self.configs["mask"][
            "negative_mask_val"
        ]  # This will be necessary to set points outside map to the minimum

        inside_country_values = self.stacked_raster_coverages[
            :, self.inside_mask_idx[0], self.inside_mask_idx[1]
        ].T

        inside_country_values_scaled = self.min_max_scaler.scale_coverages(
            inside_country_values
        )
        predicted_anomaly_detection = self.ocsvm.predict(inside_country_values_scaled)
        Z[
            self.inside_mask_idx[0], self.inside_mask_idx[1]
        ] = predicted_anomaly_detection
        self.Z = Z

    def generate(self, number_pseudo_absenses: int):
        pseudo_absenses_df = pd.DataFrame(columns=self.columns)
        # Save Z because takes too long to run
        x, y = np.where(self.Z == -1)
        coord_chosed = []
        for _ in range(number_pseudo_absenses):
            while True:
                random_val = np.random.randint(len(x))
                matrix_position = (x[random_val], y[random_val])
                coord = self._find_coordinate_in_map_matrix(matrix_position)
                if coord not in coord_chosed:
                    coord_chosed.append(coord)
                    break
            row_values = self.stacked_raster_coverages[:, x[random_val], y[random_val]]
            pseudo_absense_row = dict(zip(self.columns, row_values))
            pseudo_absenses_df = pseudo_absenses_df.append(
                pseudo_absense_row, ignore_index=True
            )

        coordinates_df = pd.DataFrame(np.array(coord_chosed), columns=["lat", "lon"])
        return pseudo_absenses_df, coordinates_df

    def get_psa_decision_map(self):
        return self.Z
