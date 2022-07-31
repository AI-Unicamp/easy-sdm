import gc
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from easy_sdm.configs import configs
from easy_sdm.raster_processing import RasterInfoExtractor
from easy_sdm.species_collection import SpeciesInfoExtractor
from easy_sdm.utils import RasterLoader


class SpeciesEnveriomentExtractor:
    """[This class extracts species information trought a set of raster layers that represent enverionmental conditions]"""

    def __init__(self):

        # Species related arguments
        self.__species_info_extractor = None
        self.__species_geodataframe = None

        # Envs related arguments
        self.__raster_info_extractor = None
        self.__raster = None

    def set_env_layer(self, raster: rasterio.io.DatasetReader):
        self.__raster_info_extractor = RasterInfoExtractor(raster)
        self.__raster = raster

    def set_species(self, species_geodataframe: gpd.GeoDataFrame):
        self.__species_info_extractor = SpeciesInfoExtractor(species_geodataframe)
        self.__species_geodataframe = species_geodataframe

    def __reset_vals(self):
        self.__species_info_extractor = None
        self.__species_geodataframe = None
        self.__raster_info_extractor = None
        self.__raster = None

    def __existance_verifier(self):
        if self.__species_geodataframe is None:
            raise ValueError("Call set_species before extract")
        if self.__raster is None:
            raise ValueError("Call set_env_layer before extract")

    def __approximate_no_data_pixels(self, raster_occurrences_array: np.ndarray):
        """[Repair problems on points very near to the the country boarders make the point change position in the raster center direction]

        Args:
            raster_occurrences_array (np.ndarray): [description]

        Raises:
            KeyError: [description]

        Returns:
            [type]: [description]
        """

        raster_array = self.__raster_info_extractor.get_array()
        resolution = self.__raster_info_extractor.get_resolution()

        species_longitudes = self.__species_info_extractor.get_longitudes()
        species_latitudes = self.__species_info_extractor.get_latitudes()

        ix = self.__calc_species_pos_x()
        iy = self.__calc_species_pos_y()

        # SpeciesInRasterPlotter.plot_all_points(raster_array,ix,iy)

        for i, (pixel_value, long, lat) in enumerate(
            zip(raster_occurrences_array, species_longitudes, species_latitudes)
        ):

            state = "diagonal"

            if pixel_value == configs["maps"]["no_data_val"]:
                incx, incy = 0, 0
                k = 0
                while pixel_value == configs["maps"]["no_data_val"]:

                    if k % 50 == 0:
                        if state == "diagonal":
                            state = "horizontal"
                        elif state == "horizontal":
                            state = "vertical"
                        elif state == "vertical":
                            state == "diagonal"

                    if k == 200:
                        raise KeyError(
                            "Probably there is problem in the raster once it could not find a valid value"
                        )
                    # walking coodinates in center map
                    if state == "diagonal":
                        if (
                            long >= self.__raster_info_extractor.get_xcenter()
                            and lat >= self.__raster_info_extractor.get_ycenter()
                        ):
                            long -= resolution
                            lat -= resolution
                            incx -= 1
                            incy -= 1
                        elif (
                            long >= self.__raster_info_extractor.get_xcenter()
                            and lat <= self.__raster_info_extractor.get_ycenter()
                        ):
                            long -= resolution
                            lat += resolution
                            incx -= 1
                            incy += 1
                        elif (
                            long <= self.__raster_info_extractor.get_xcenter()
                            and lat <= self.__raster_info_extractor.get_ycenter()
                        ):
                            long += resolution
                            lat += resolution
                            incx += 1
                            incy += 1
                        elif (
                            long <= self.__raster_info_extractor.get_xcenter()
                            and lat >= self.__raster_info_extractor.get_ycenter()
                        ):
                            long += resolution
                            lat -= resolution
                            incx += 1
                            incy -= 1
                    elif state == "horizontal":
                        if long <= self.__raster_info_extractor.get_xcenter():
                            long += resolution
                            incx += 1
                        else:
                            long -= resolution
                            incx -= 1
                    elif state == "vertical":
                        if lat <= self.__raster_info_extractor.get_ycenter():
                            lat += resolution
                            incy += 1
                        else:
                            lat -= resolution
                            incy -= 1

                    newx_point = ix[i] + incx
                    newy_point = iy[i] + incy

                    pixel_value = raster_array[-newy_point, newx_point]

                    if pixel_value != configs["maps"]["no_data_val"]:
                        raster_occurrences_array[i] = pixel_value

                    k += 1

        return raster_occurrences_array

    def __calc_species_pos_x(self):
        return np.searchsorted(
            self.__raster_info_extractor.get_xgrid(),
            self.__species_info_extractor.get_longitudes(),
        )

    def __calc_species_pos_y(self):
        return np.searchsorted(
            self.__raster_info_extractor.get_ygrid(),
            self.__species_info_extractor.get_latitudes(),
        )

    def extract(self):

        # Check if required fields were set
        self.__existance_verifier()

        # Extract enverionment values for coordinates
        ix = self.__calc_species_pos_x()
        iy = self.__calc_species_pos_y()
        raster_array = self.__raster_info_extractor.get_array()
        raster_occurrences_array = raster_array[-iy, ix]
        del raster_array
        del ix
        del iy

        # Treat with no data values
        raster_occurrences_array = self.__approximate_no_data_pixels(
            raster_occurrences_array
        )
        gc.collect()

        # Reset vals for the next call
        self.__reset_vals()

        return raster_occurrences_array


class OccurrancesDatasetBuilder:
    def __init__(self, raster_path_list: List[Path]):
        self.raster_path_list = raster_path_list
        self.species_env_extractor = SpeciesEnveriomentExtractor()

    def __get_var_names(self):
        return [Path(path).name.split(".")[0] for path in self.raster_path_list]

    def build(
        self, species_gdf: gpd.GeoDataFrame,
    ):
        """Save all extracted to a numpy array"""

        all_env_values_species_list = []
        for path in self.raster_path_list:
            raster = RasterLoader(path).load_dataset()
            self.species_env_extractor.set_env_layer(raster)
            self.species_env_extractor.set_species(species_gdf)
            raster_occurrences_array = self.species_env_extractor.extract()
            all_env_values_species_list.append(raster_occurrences_array)
            del raster
            del raster_occurrences_array
            gc.collect()

        occurrances_df = pd.DataFrame(
            np.vstack(all_env_values_species_list).T, columns=self.__get_var_names()
        )
        occurrances_df["label"] = 1
        self.dataset = occurrances_df
        coordinates_df = pd.DataFrame(
            SpeciesInfoExtractor(species_gdf).get_coordinates(), columns=["lat", "lon"]
        )
        self.coordinates_df = coordinates_df

    def get_dataset(self):
        return self.dataset

    def get_coordinates_df(self):
        return self.coordinates_df
