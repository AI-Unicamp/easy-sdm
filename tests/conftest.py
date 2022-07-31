from pathlib import Path

import pytest

from easy_sdm.typos import Species


@pytest.fixture
def mock_species():
    return Species(taxon_key=7587087, name="Cajanus cajan")


############################
#           Paths          #
############################

# @pytest.fixture
# def raw_rasters_dirpath():
#     raw_rasters_dirpath = PathUtils.dir_path(Path.cwd() / "data/download/raw_rasters")
#     return raw_rasters_dirpath


# @pytest.fixture
# def processed_rasters_dirpath():
#     processed_rasters_dirpath = PathUtils.dir_path(
#         Path.cwd() / data/raster_processing/environment_variables_rasters
#     )
#     return processed_rasters_dirpath


@pytest.fixture
def root_test_data_path():
    return Path.cwd() / "mock_data"


@pytest.fixture
def mock_species_shapefile_path(root_test_data_path):
    path = root_test_data_path / "species_collection/cajanus_cajan"
    return path


@pytest.fixture
def mock_map_shapefile_path(root_test_data_path):
    path = root_test_data_path / "download/region_shapefile"
    return path


@pytest.fixture
def mock_raw_raster_path_bioclim(root_test_data_path):
    path = (
        root_test_data_path
        / "download/raw_rasters/Bioclim/bio1_annual_mean_temperature.tif"
    )
    return path


@pytest.fixture
def mock_raw_raster_path_soilgrids(root_test_data_path):
    path = root_test_data_path / "download/raw_rasters/Soilgrids/clay_0-5cm_mean.tif"
    return path


@pytest.fixture
def mock_processed_raster_path_bioclim(root_test_data_path):
    path = (
        root_test_data_path
        / "raster_processing/environment_variables_rasters/Bioclim/bio1_annual_mean_temperature.tif"
    )
    return path


@pytest.fixture
def mock_processed_raster_path_soilgrids(root_test_data_path):
    path = (
        root_test_data_path
        / "raster_processing/environment_variables_rasters/Soilgrids/clay_0-5cm_mean.tif"
    )
    return path


@pytest.fixture
def mock_mask_raster_path(root_test_data_path):
    path = root_test_data_path / "raster_processing/region_mask.tif"
    return path


@pytest.fixture
def mock_environment_dirpath(root_test_data_path):
    path = root_test_data_path / "environment"
    return path


@pytest.fixture
def mock_featuarizer_dirpath(root_test_data_path):
    path = root_test_data_path / "featuarizer"
    return path


############################
#           Lists          #
############################


@pytest.fixture
def processed_raster_paths_list(
    mock_processed_raster_path_bioclim, mock_processed_raster_path_soilgrids
):
    raster_paths_list = [
        mock_processed_raster_path_bioclim,
        mock_processed_raster_path_soilgrids,
    ]
    return raster_paths_list


@pytest.fixture
def raw_raster_paths_list(mock_raw_raster_path_bioclim, mock_raw_raster_path_soilgrids):
    raster_paths_list = [mock_raw_raster_path_bioclim, mock_raw_raster_path_soilgrids]
    return raster_paths_list


############################
#           Special          #
############################

# O certo seria deixar isso aqui e criar um MockRasterStatisticsCalculator
# @pytest.fixture
# def df_stats(tmp_path, mock_mask_raster_path,processed_raster_paths_list):

#     from easy_sdm.dataset_creation import RasterStatisticsCalculator

#     output_path = tmp_path / "rasters_statistics.csv"
#     RasterStatisticsCalculator(
#         raster_path_list=processed_raster_paths_list, mask_raster_path=mock_mask_raster_path
#     ).build_table(output_path)

#     df_stats = pd.read_csv(output_path)
#     df_stats.to_csv('extras/rasters_statistics.csv',index=False)
#     return df_stats
