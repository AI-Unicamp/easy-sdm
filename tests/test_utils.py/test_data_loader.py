import geopandas as gpd
import rasterio

from easy_sdm.utils import RasterLoader, ShapefileLoader


def test_shapefile_loader(mock_map_shapefile_path):
    species_gdf = ShapefileLoader(mock_map_shapefile_path).load_dataset()

    assert isinstance(species_gdf, gpd.GeoDataFrame)
    assert "geometry" in species_gdf.columns
    assert species_gdf.shape[0] > 0


def test_raster_loader(mock_processed_raster_path_bioclim):
    raster = RasterLoader(mock_processed_raster_path_bioclim).load_dataset()

    assert isinstance(raster, rasterio.io.DatasetReader)
