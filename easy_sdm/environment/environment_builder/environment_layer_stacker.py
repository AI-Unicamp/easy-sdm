import gc
from pathlib import Path
from typing import List

import numpy as np

from easy_sdm.raster_processing import RasterInfoExtractor
from easy_sdm.raster_processing.processing.raster_information_extractor import (
    RasterInfoExtractor,
)
from easy_sdm.utils import RasterLoader


class EnverionmentLayersStacker:
    """[Create a 3D array from 2D arrays stacked]"""

    def __init__(self) -> None:
        pass

    def load(self, input_path: Path):
        return np.load(input_path)

    def stack_and_save(self, raster_path_list: List[Path], output_path: Path):
        assert str(output_path).endswith(".npy"), "output_path must ends with .npy"

        coverage = self.stack(raster_path_list)
        with open(output_path, "wb") as f:
            np.save(f, coverage)

        del coverage
        gc.collect()

    def stack(self, raster_path_list: List[Path]):

        all_env_values_list = []
        for path in raster_path_list:
            raster = RasterLoader(path).load_dataset()
            raster_array = RasterInfoExtractor(raster).get_array()
            all_env_values_list.append(raster_array)
            del raster
            del raster_array
            gc.collect()

        # [vec.shape for vec in all_env_values_list]
        coverage = np.stack(all_env_values_list)

        return coverage
