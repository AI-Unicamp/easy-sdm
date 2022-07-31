import geopandas as gpd
import numpy as np


class SpeciesInfoExtractor:
    """[A Wrapper to extract relevant information from spescies geodataframes]"""

    def __init__(self, species_geodataframe: gpd.GeoDataFrame) -> None:
        self.species_geodataframe = species_geodataframe

    def get_coordinates(self,):
        coordinates = np.array(
            (
                np.array(self.species_geodataframe["LATITUDE"]),
                np.array(self.species_geodataframe["LONGITUDE"]),
            )
        ).T
        return coordinates

    def get_longitudes(self,):
        coordinates = self.get_coordinates()
        return coordinates[:, 1]

    def get_latitudes(self,):
        coordinates = self.get_coordinates()
        return coordinates[:, 0]
