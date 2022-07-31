import pickle
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from easy_sdm.configs import configs


class RasterLoader:
    def __init__(self, raster_path: Path):
        self.configs = configs
        self.raster_path = raster_path

    def load_dataset(self):
        raster = rasterio.open(self.raster_path)
        return raster


class ShapefileLoader:
    def __init__(self, shapefile_path: Path):
        self.shapefile_path = shapefile_path

    def __get_shp_file(self):
        shp_file_path = self.shapefile_path
        if not str(self.shapefile_path).endswith(".shp"):
            shp_file_path = [f for f in self.shapefile_path.glob("*.shp")][0]
        return shp_file_path

    def load_dataset(self):
        shp_file_path = self.__get_shp_file()
        shp = gpd.read_file(shp_file_path)
        return shp


class NumpyArrayLoader:
    def __init__(self, dataset_path: str,) -> None:
        self.dataset_path = dataset_path

    def load_dataset(self):

        with open(self.dataset_path, "rb") as f:
            np_array = np.load(f)
        return np_array


class PickleLoader:
    def __init__(self, dataset_path: str,) -> None:
        self.dataset_path = dataset_path

    def load_dataset(self):

        with open(self.dataset_path, "rb") as fp:  # Unpickling
            raster_list = pickle.load(fp)

        return raster_list


class DatasetLoader:
    def __init__(
        self, dataset_path: str, output_column: str = "", output_format: str = "pandas"
    ) -> None:
        self.dataset_path = dataset_path
        self.output_column = output_column
        self.output_format = output_format

    def load_dataset(self):

        df = pd.read_csv(self.dataset_path)
        feature_columns = [i for i in df.columns if i not in self.output_column]
        x = df.filter(feature_columns)
        y = df.filter([self.output_column])

        if self.output_format == "numpy":
            x = x.to_numpy()
            y = y.to_numpy()

        return x, y
