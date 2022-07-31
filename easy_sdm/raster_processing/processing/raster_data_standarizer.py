from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.interpolate import NearestNDInterpolator

from easy_sdm.configs import configs
from easy_sdm.enums import RasterSource
from easy_sdm.utils import RasterUtils
from easy_sdm.utils.data_loader import RasterLoader


class RasterMissingValueFiller:
    """solution based in:
    https://stackoverflow.com/questions/68197762/fill-nan-with-nearest-neighbor-in-numpy-array
    """

    def __init__(self) -> None:
        pass

    def fill_missing_values(self, data: np.ndarray, no_value_list: List[float]):
        for no_val in no_value_list:
            data[data == no_val] = np.nan

        mask = np.where(~np.isnan(data))
        interp = NearestNDInterpolator(np.transpose(mask), data[mask])
        filled_data = interp(*np.indices(data.shape))

        return filled_data


class RasterDataStandarizer:
    """[A class to perform expected standarizations]"""

    def __init__(self, data_dirpath) -> None:
        self.configs = configs
        self.data_dirpath = data_dirpath
        self.region_mask_array = (
            RasterLoader(data_dirpath / "raster_processing/region_mask.tif")
            .load_dataset()
            .read(1)
        )
        self.inputer = RasterMissingValueFiller()

    def __one_step_standarization(self, data: np.ndarray):
        data = np.where(
            self.region_mask_array == configs["maps"]["no_data_val"],
            configs["maps"]["no_data_val"],
            self.inputer.fill_missing_values(data, [configs["maps"]["no_data_val"]]),
        )
        return data

    def __standarize_country_borders(self, data: np.ndarray):

        data = np.where(
            self.region_mask_array == configs["maps"]["no_data_val"],
            configs["maps"]["no_data_val"],
            data,
        )

        return data

    def __standarize_no_data_val(self, profile: Dict, data: np.ndarray):

        data = np.where(data == profile["nodata"], configs["maps"]["no_data_val"], data)

        return data

    def __assert_standarizarion_is_correct(self, raster_array):
        raster_array = np.where(
            self.region_mask_array == configs["maps"]["no_data_val"],
            -1000,
            raster_array,
        )

        assert raster_array.min() == -1000

    def __set_soilgrids_no_data_to_profile(self, profile):
        profile["nodata"] = 0
        return profile

    def standarize(self, raster, raster_source: RasterSource, output_path: Path):
        profile = raster.profile.copy()
        if raster_source == RasterSource.Soilgrids:
            profile = self.__set_soilgrids_no_data_to_profile(profile)
        data = raster.read(1)
        height, width = data.shape
        data = np.float32(data)

        data = self.__standarize_no_data_val(profile, data)
        data = self.__one_step_standarization(data=data)
        self.__assert_standarizarion_is_correct(data)

        profile.update(
            {
                "driver": "GTiff",
                "count": 1,
                "crs": {"init": f"EPSG:{configs['maps']['default_epsg']}"},
                "width": width,
                "height": height,
                "nodata": configs["maps"]["no_data_val"],
                "dtype": np.float32,
            }
        )
        RasterUtils.save_raster(data=data, profile=profile, output_path=output_path)
