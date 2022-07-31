import geopandas as gpd
import rasterio


def test_species_shapefile_dataset(mock_species_shapefile_dataloader):
    species_gdf = mock_species_shapefile_dataloader.load_dataset()

    assert isinstance(species_gdf, gpd.GeoDataFrame)
    assert "geometry" in species_gdf.columns
    assert species_gdf.shape[0] > 0


def test_map_shapefile_dataset(mock_map_shapefile_dataloader):
    map_gdf = mock_map_shapefile_dataloader.load_dataset()

    assert isinstance(map_gdf, gpd.GeoDataFrame)
    assert "geometry" in map_gdf.columns
    assert map_gdf.shape[0] > 0


def test_raw_raster_dataset(mock_raw_raster_dataloader):
    raster = mock_raw_raster_dataloader.load_dataset()

    assert isinstance(raster, rasterio.io.DatasetReader)


def test_processed_raster_dataset(mock_processed_raster_dataloader):
    raster = mock_processed_raster_dataloader.load_dataset()

    assert isinstance(raster, rasterio.io.DatasetReader)
