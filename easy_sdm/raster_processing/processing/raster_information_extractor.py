import numpy as np
import rasterio

from easy_sdm.configs import configs


class RasterInfoExtractor:
    """[A Wrapper to extract relevant information from raster objects]"""

    def __init__(self, raster: rasterio.io.DatasetReader):
        self.configs = configs
        self.__raster_array = self.__extrac_array(raster)
        self.__meta = self.__extract_meta(raster)

    def __extrac_array(self, raster: rasterio.io.DatasetReader):
        self.__one_layer_verifier(raster)
        raster_array = raster.read(1)
        return raster_array

    def __one_layer_verifier(self, raster: rasterio.io.DatasetReader):
        if raster.meta["count"] > 1:
            raise ValueError("Raster images are suppose to have only one layer")
        elif raster.meta["count"] == 0:
            raise ValueError("For some reason this raster is empty")

    def __extract_meta(self, raster: rasterio.io.DatasetReader):
        return raster.meta

    def get_array(self):
        return self.__raster_array

    def get_driver(self):
        return self.__meta["GTiff"]

    def get_data_type(self):
        return self.__meta["dtype"]

    def get_nodata_val(self):
        return self.__meta["nodata"]

    def get_width(self):
        return self.__meta["width"]

    def get_heigh(self):
        return self.__meta["heigh"]

    def get_crs(self):
        return self.__meta["crs"]

    def get_affine(self):
        return self.__meta["transform"]

    def get_resolution(self):
        return abs(self.__meta["transform"][0])

    def get_xgrid(self):
        xgrid = np.arange(
            self.configs["maps"]["region_limits_with_security"]["west"],
            self.configs["maps"]["region_limits_with_security"]["west"]
            + self.__raster_array.shape[1] * self.get_resolution(),
            self.get_resolution(),
        )
        return xgrid

    def get_ygrid(self):
        ygrid = np.arange(
            configs["maps"]["region_limits_with_security"]["south"],
            configs["maps"]["region_limits_with_security"]["south"]
            + self.__raster_array.shape[0] * self.get_resolution(),
            self.get_resolution(),
        )
        return ygrid

    def get_xcenter(self):
        xgrid = self.get_xgrid()
        return np.mean(xgrid)

    def get_ycenter(self):
        ygrid = self.get_ygrid()
        return np.mean(ygrid)
