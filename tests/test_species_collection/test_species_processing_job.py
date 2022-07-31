from pathlib import Path

import geopandas as gpd

from easy_sdm.species_collection import SpeciesCollectionJob
from easy_sdm.typos import Species
from easy_sdm.utils import ShapefileLoader


def test_species_collection_job(tmp_path, mock_species, mock_map_shapefile_path):

    job = SpeciesCollectionJob(
        output_dirpath=tmp_path, region_shapefile_path=mock_map_shapefile_path
    )
    job.collect_species_data(mock_species)

    geo_df = ShapefileLoader(
        tmp_path / mock_species.get_name_for_paths()
    ).load_dataset()
    assert "geometry" in geo_df.columns
    assert isinstance(geo_df, gpd.GeoDataFrame)
