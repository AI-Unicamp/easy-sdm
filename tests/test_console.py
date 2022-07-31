from distutils.command.config import config
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from easy_sdm.configs import configs
from easy_sdm.data_colector import burn_shapefile_in_raster, standarize_rasters
from easy_sdm.dataset_creation import (
    OccurrancesDatasetBuilder,
    RasterStatisticsCalculator,
    SDMDatasetCreator,
)
from easy_sdm.raster_processing import (
    RasterInfoExtractor,
    RasterLoader,
    RasterStandarizer,
    ShapefileLoader,
)
from easy_sdm.utils import PathUtils


def test_standarize_rasters(tmp_path, raw_rasters_dirpath):

    standarize_rasters(
        source_dirpath=raw_rasters_dirpath / "Elevation_Rasters",
        destination_dirpath=tmp_path,
        raster_type="elevation",
    )
    raster1 = tmp_path / "elev1_strm_worldclim_elevation.tif"
    raster2 = tmp_path / "elev2_envirem_terrain_roughness_index.tif"
    assert Path(raster1).is_file()
    assert Path(raster2).is_file()
    assert isinstance(RasterLoader(raster1).load_dataset(), rasterio.io.DatasetReader)


def test_burn_shapefile_in_raster(
    tmp_path, mock_map_shapefile_path, mock_processed_raster_path
):
    output_path = tmp_path / "region_mask.tif"
    burn_shapefile_in_raster(
        reference_raster_path=mock_processed_raster_path,
        shapefile_path=mock_map_shapefile_path,
        output_path=output_path,
    )

    raster_mask = RasterLoader(output_path).load_dataset()
    raster_mask_array = RasterInfoExtractor(raster_mask).get_array()
    array_01 = np.array([-9999.0, 0])
    uniques = np.unique(raster_mask_array)
    assert np.array_equal(uniques, array_01)


def test_statistics_table_generatio(tmp_path, mock_master_raster_path):
    processed_rasters_dirpath = PathUtils.dir_path(
        Path.cwd() / "data/processed_rasters/standarized_rasters"
    )
    raster_path_list = PathUtils.get_rasters_filepaths_in_dir(processed_rasters_dirpath)

    output_path = tmp_path / "rasters_statistics.csv"
    RasterStatisticsCalculator(
        raster_path_list=raster_path_list, mask_raster_path=mock_master_raster_path
    ).build_table(output_path)

    df_stats = pd.read_csv(output_path)
    assert df_stats.shape(0) == len(raster_path_list)

    # def test_sdm_dataset_creator(mock_species_shapefile_path):
    #     processed_rasters_dirpath = PathUtils.dir_path(Path.cwd() / "data/processed_rasters/standarized_rasters")
    #     species_shapefile_path = PathUtils.file_path(mock_species_shapefile_path)
    #     SDMDatasetCreator(raster_path_list=, statistics_dataset=)

    #     class SDMDatasetCreator:
    #     """[Create a dataset with species and pseudo spescies for SDM Machine Learning]
    #     """

    #     def __init__(
    #         self, raster_path_list: List[Path], statistics_dataset: pd.DataFrame
    #     ) -> None:
    #         # ps_generator: BasePseudoSpeciesGenerator
    #         # self.ps_generator = ps_generator
    #         self.statistics_dataset = statistics_dataset
    #         self.occ_dataset_builder = OccurrancesDatasetBuilder(raster_path_list)

    # raster_paths_list = PathUtils.get_rasters_filepaths_in_dir(
    #     processed_rasters_dirpath
    # )
    # occ_dst_builder = OccurrancesDatasetBuilder(raster_paths_list)
    # df = occ_dst_builder.build(
    #     ShapefileLoader(species_shapefile_path).load_dataset()
    # )

    # df.to_csv(Path.cwd() / 'extras' / "mock_occurrences.csv" ,index=False)

    SDMDatasetCreator()
