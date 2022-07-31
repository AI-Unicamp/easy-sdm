import pickle
from pathlib import Path
from typing import List


class RelevantRastersSelector:
    def __init__(self) -> None:
        pass

    def get_relevant_raster_path_list(self, raster_path_list: List[Path]):
        raster_path_list = self.__filter_by_root_size(raster_path_list)
        raster_path_list = self.__get_not_colinear_raster_path_list(raster_path_list)
        return raster_path_list

    def __filter_by_root_size(self, raster_path_list: List[Path]):
        # TODO:
        return raster_path_list

    def __get_not_colinear_raster_path_list(self, raster_path_list: List[Path]):
        return raster_path_list

    def save_raster_list(self, raster_path_list: List[Path], output_path: Path):
        with open(output_path, "wb") as fp:  # Pickling
            pickle.dump(raster_path_list, fp)
