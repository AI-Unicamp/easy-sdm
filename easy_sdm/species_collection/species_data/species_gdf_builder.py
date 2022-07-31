from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import pandas as pd
import requests

from easy_sdm.configs import configs
from easy_sdm.typos import Species
from easy_sdm.utils import logger

from .species_in_shapefile_checker import SpeciesInShapefileChecker


class GBIFOccurencesRequester:
    """[This class makes request to GBIF]"""

    def __init__(self, species: Species):

        self.species = species
        self.base_url = "http://api.gbif.org/v1/occurrence/search"

    def request(self, offset: int = 0):
        """[ Request GBIF information about an species]

        Args:
            offset (int, optional): [Offsset is a parameter to where starting the
            request in GBIF databse, since the requests have a
            limit of 300 row for request]. Defaults to 0.

        Returns:
            [type]: [int]
        """

        gbif_configs = configs["gbif"]
        params = {
            "taxonKey": str(self.species.taxon_key),
            "limit": gbif_configs["one_request_limit"],
            "hasCoordinate": True,
            "year": f"{gbif_configs['low_year']},{gbif_configs['up_year']}",
            "country": gbif_configs["country"],
            "offset": offset,
        }
        r = requests.get(self.base_url, params=params)
        status_code = r.status_code
        if r.status_code != 200:
            logger.logging.info(
                f"API call failed at offset {offset} with a status code of {r.status_code}."
            )
            end_of_records = True
        else:
            r = r.json()
            end_of_records = r["endOfRecords"]

        return r, end_of_records, status_code


class SpeciesDFBuilder:
    """[This class organize data requested to GBIF into pandas dataframes]"""

    def __init__(self, species: Species):
        self.gbif_occ_requester = GBIFOccurencesRequester(species=species)
        self.__df_memory = None

    def get_specie_df(self):
        """Get species as DataFrame"""
        if self.__df_memory:
            df = self.__df_memory
        else:
            df = self.__request_species_df()
            df = self.__clean_species_df(df)
            self.__df_memory = df
        return df

    def __request_species_df(self):
        """[Organizes GBIF information in a dataframe considering offsets ]"""

        end_of_records = False
        offset = 0
        status = 200
        df = None
        while end_of_records == False and status == 200:
            r, end_of_records, status = self.gbif_occ_requester.request(offset)
            df = self.__build_species_df(r, df)
            offset = len(df) + 1

        self.__clean_species_df(df)
        return df

    def __build_species_df(self, request, df=None):
        """[Create species dataframe with the request data]

        Args:
            df ([type]): [description]
            request ([type]): [description]

        Returns:
            [df]: [description]
        """
        if df is None:
            df = pd.DataFrame(
                columns=[
                    "SCIENTIFIC_NAME",
                    "LONGITUDE",
                    "LATITUDE",
                    "COUNTRY",
                    "STATE_PROVINCE",
                    "IDENTIFICATION_DATE",
                    "DAY",
                    "MONTH",
                    "YEAR",
                ]
            )

        for result in request["results"]:
            result = self.__refact_dict(result)
            df = df.append(
                {
                    "SCIENTIFIC_NAME": result["scientificName"],
                    "LONGITUDE": result["decimalLongitude"],
                    "LATITUDE": result["decimalLatitude"],
                    "COUNTRY": result["country"],
                    "STATE_PROVINCE": result["stateProvince"],
                    "IDENTIFICATION_DATE": result["eventDate"],
                    "DAY": result["day"],
                    "MONTH": result["month"],
                    "YEAR": result["year"],
                },
                ignore_index=True,
            )
        return df

    def __refact_dict(self, result: Dict):
        """Refact dict placing None in empty cells"""

        columns = result.keys()
        desired_columns = [
            "scientificName",
            "decimalLongitude",
            "decimalLatitude",
            "country",
            "stateProvince",
            "eventDate",
            "day",
            "month",
            "year",
            "occurrenceRemarks",
        ]
        for d_col in desired_columns:
            if d_col not in columns:
                result[d_col] = None
        return result

    def __clean_species_df(self, df: pd.DataFrame):
        """[Cleaning Gbif Data]

        Args:
            df ([pd.DaraFrame]): [description]

        Returns:
            [pd.DaraFrame]: [description]
        """
        # Double check to certify there is no empty lat/long data
        df = df[pd.notnull(df["LATITUDE"])]
        df = df[pd.notnull(df["LONGITUDE"])]

        # Removing duplicate data
        df = (
            df.drop_duplicates(ignore_index=True)
            if configs["gbif"]["drop_duplicates"]
            else df
        )

        # Sorting Data by STATE_PROVINCE
        df.sort_values("STATE_PROVINCE", inplace=True, ignore_index=True)
        return df


class SpeciesGDFBuilder(SpeciesDFBuilder):
    """[This class organize data requested to GBIF into geopandas geodataframes]"""

    def __init__(
        self, species: Species, proposed_region: Optional[gpd.GeoDataFrame] = None,
    ):
        super().__init__(species)
        self.proposed_region = proposed_region
        self.__gdf_memory = None

    def save_species_gdf(self, output_path: Path):
        if not str(output_path).endswith(".shp"):
            raise TypeError("output_path must ends with shp")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gdf = self.get_species_gdf()
        gdf.to_file(output_path)

    def get_species_gdf(self):
        if not (self.__gdf_memory is None):
            gdf = self.__gdf_memory
        else:
            df = self.get_specie_df()
            gdf = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE)
            )
            gdf = gdf.set_crs(f"EPSG:{configs['maps']['default_epsg']}")
            gdf = (
                self.__filter_species_in_region(gdf)
                if not (self.proposed_region is None)
                else gdf
            )
            self.__gdf_memory = gdf
        return gdf

    def __filter_species_in_region(self, gdf: gpd.GeoDataFrame):
        return SpeciesInShapefileChecker(self.proposed_region).get_points_inside(gdf)
