from pathlib import Path

import numpy as np
import rasterio

from easy_sdm.configs import configs
from easy_sdm.utils import RasterUtils


class RasterCliper:
    """
    [Clip a raster trought a configured square]

    OBS: After saved metadata weight and width are inverted to fit
    with the array shape. It is not possible to do this inversion before.
    So keep in mind this detail. For the hole process there will not have
    any problem after this step

    """

    def __get_window_from_extent(self, aff):
        """Get a portion form a raster array based on the country limits"""
        map_limits = configs["maps"]["region_limits_with_security"]
        col_start, row_start = ~aff * (map_limits["west"], map_limits["north"],)
        col_stop, row_stop = ~aff * (map_limits["east"], map_limits["south"],)
        return ((int(row_start), int(row_stop)), (int(col_start), int(col_stop)))

    def __create_profile(
        self,
        source_raster: rasterio.io.DatasetReader,
        cropped_raster_matrix: np.ndarray,
    ):
        """[Create profile with required changes to clip rasters]
        Affine logic: (resolution_x,rot1,extreme_west_point,rot2,resolution_y(negative),extreme_north_point)
        Example: Affine(0.008333333333333333, 0.0, -180.0, 0.0, -0.008333333333333333, 90.0)
        """
        map_limits = configs["maps"]["region_limits_with_security"]
        result_profile = source_raster.profile.copy()
        cropped_data = cropped_raster_matrix.copy()
        src_transfrom = result_profile["transform"]
        result_transfrom = rasterio.Affine(
            src_transfrom[0],
            src_transfrom[1],
            map_limits["west"],
            src_transfrom[3],
            src_transfrom[4],
            map_limits["north"],
        )

        # width > height (Brasil)
        result_profile.update(
            {
                "width": cropped_data.shape[1],
                "height": cropped_data.shape[0],
                "transform": result_transfrom,
            }
        )

        return result_profile

    def clip(self, source_raster: rasterio.io.DatasetReader):
        window_region = self.__get_window_from_extent(source_raster.meta["transform"])
        cropped_raster_matrix = source_raster.read(1, window=window_region)
        profile = self.__create_profile(source_raster, cropped_raster_matrix)
        return profile, cropped_raster_matrix

    def clip_and_save(
        self, source_raster: rasterio.io.DatasetReader, output_path: Path
    ):
        result_profile, cropped_raster_matrix = self.clip(source_raster)
        RasterUtils.save_raster(
            data=cropped_raster_matrix, profile=result_profile, output_path=output_path
        )
