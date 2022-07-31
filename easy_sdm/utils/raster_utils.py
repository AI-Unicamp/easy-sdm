from pathlib import Path
from typing import Dict

import numpy as np
import rasterio


class RasterUtils:
    def __init__(self):
        pass

    @classmethod
    def save_raster(cls, data: np.ndarray, profile: Dict, output_path: Path):

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)

    @classmethod
    def save_binary_raster(cls, binary_raster: str, output_path: Path):

        with open(output_path, "wb") as file:
            file.write(binary_raster.read())
