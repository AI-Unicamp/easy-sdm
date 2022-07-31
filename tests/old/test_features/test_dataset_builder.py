from pathlib import Path

from easy_sdm.featuarizer import OccurrancesDatasetBuilder
from easy_sdm.utils import PathUtils, ShapefileLoader

# def test_extract_occurances(mock_species_shapefile_path, processed_raster_paths_list):

#     species_shapefile_path = PathUtils.file_path(mock_species_shapefile_path)
#     occ_dst_builder = OccurrancesDatasetBuilder(processed_raster_paths_list)
#     df = occ_dst_builder.build(ShapefileLoader(species_shapefile_path).load_dataset())
#     df.to_csv("extras/mock_occurrences.csv")
#     assert(df.empty is False)
#     assert(df.index.names == ['lat', 'lon'])


def test_extract_occurances(mock_species_shapefile_path, processed_raster_paths_list):

    species_shapefile_path = PathUtils.file_path(mock_species_shapefile_path)
    occ_dst_builder = OccurrancesDatasetBuilder(processed_raster_paths_list)
    df = occ_dst_builder.build(ShapefileLoader(species_shapefile_path).load_dataset())
    assert df.empty is False
    assert df.index.names == ["lat", "lon"]
