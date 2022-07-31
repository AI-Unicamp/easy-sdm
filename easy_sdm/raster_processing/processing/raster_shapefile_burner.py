from pathlib import Path

import geopandas as gpd
import rasterio
import rasterio.features


class RasterShapefileBurner:
    """[Burn a shapefile troought a reference raster. The reference is used only to get the Affine properties]
    Inside Shapefile: 0
    Outside Shapefile: -9999. (no_data_val)
    """

    def __init__(self, reference_raster: rasterio.io.DatasetReader):
        meta = reference_raster.meta.copy()
        meta.update(compress="lzw")
        self.meta = meta

    def burn_and_save(self, shapfile: gpd.GeoDataFrame, output_path: Path):
        """Inspired for #https://gis.stackexchange.com/questions/151339/rasterize-a-shapefile-with-geopandas-or-fiona-python

        Args:
            shapfile (gpd.GeoDataFrame): [description]
            output_path (str): [description]
        """

        with rasterio.open(output_path, "w+", **self.meta) as out:
            out_arr = out.read(1)

            # this is where we create a generator of geom, value pairs to use in rasterizing
            shapes = ((geom, 0) for geom in shapfile.geometry)

            burned = rasterio.features.rasterize(
                shapes=shapes,
                fill=0,
                out=out_arr,
                all_touched=True,
                transform=out.transform,
            )

            out.write_band(1, burned)
