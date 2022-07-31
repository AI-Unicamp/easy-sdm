import geopandas as gpd

from easy_sdm.species_collection import SpeciesGDFBuilder, SpeciesInShapefileChecker
from easy_sdm.typos import Species
from easy_sdm.utils import ShapefileLoader


def test_species_inside_regions(
    tmp_path, mock_map_shapefile_path, mock_species_shapefile_path
):

    mock_species_gdf = ShapefileLoader(mock_species_shapefile_path).load_dataset()
    shp_region = ShapefileLoader(mock_map_shapefile_path).load_dataset()
    checker = SpeciesInShapefileChecker(shp_region)
    new_mock_species_gdf = checker.get_points_inside(mock_species_gdf)
    assert type(new_mock_species_gdf) is gpd.GeoDataFrame
    assert len(new_mock_species_gdf) <= len(mock_species_gdf)
