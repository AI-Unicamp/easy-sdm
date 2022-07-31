from pathlib import Path
from typing import List

from easy_sdm.configs import configs
from easy_sdm.utils import PathUtils

from .environment_builder.environment_layer_stacker import EnverionmentLayersStacker
from .environment_builder.relevant_rasters_selector import RelevantRastersSelector
from .environment_builder.statistics_calculator import RasterStatisticsCalculator


class EnvironmentCreationJob:
    def __init__(self, data_dirpath: Path, all_rasters_path_list: List[Path]) -> None:

        self.data_dirpath = data_dirpath
        self.all_rasters_path_list = all_rasters_path_list
        self.relevant_rasters_selector = RelevantRastersSelector()
        self.env_layer_stacker = EnverionmentLayersStacker()

    def __build_empty_folders(self, path):
        PathUtils.create_folder(path)

    def build_environment(self):

        output_dirpath = self.data_dirpath / "environment"
        self.__build_empty_folders(output_dirpath)

        relevant_raster_path_list = self.relevant_rasters_selector.get_relevant_raster_path_list(
            raster_path_list=self.all_rasters_path_list,
        )

        self.relevant_rasters_selector.save_raster_list(
            raster_path_list=relevant_raster_path_list,
            output_path=output_dirpath / "relevant_raster_list",
        )

        self.env_layer_stacker.stack_and_save(
            raster_path_list=relevant_raster_path_list,
            output_path=output_dirpath / "environment_stack.npy",
        )

        region_mask_raster_path = (
            self.data_dirpath / "raster_processing" / "region_mask.tif"
        )
        raster_statistics_calculator = RasterStatisticsCalculator(
            raster_path_list=relevant_raster_path_list,
            mask_raster_path=region_mask_raster_path,
        )
        raster_statistics_calculator.build_table(
            output_dirpath / "raster_statistics.csv"
        )
