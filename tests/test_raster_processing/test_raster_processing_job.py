import rasterio

from easy_sdm.enums import RasterSource
from easy_sdm.raster_processing import RasterProcessingJob
from easy_sdm.utils import RasterLoader


def test_raster_processing_job(root_test_data_path):

    processed_rasters_dir = root_test_data_path / "raster_processing"

    raster_processing_job = RasterProcessingJob(data_dirpath=root_test_data_path)

    raster_processing_job.process_rasters_from_source(RasterSource.Bioclim)
    raster_processing_job.process_rasters_from_source(RasterSource.Soilgrids)

    processed_raster_bioclim = RasterLoader(
        processed_rasters_dir
        / "environment_variables_rasters"
        / RasterSource.Bioclim.name
        / "bio1_annual_mean_temperature.tif"
    ).load_dataset()
    processed_raster_soilgrids = RasterLoader(
        processed_rasters_dir
        / "environment_variables_rasters"
        / RasterSource.Soilgrids.name
        / "clay_0-5cm_mean.tif"
    ).load_dataset()
    region_mask = RasterLoader(processed_rasters_dir / "region_mask.tif").load_dataset()
    assert isinstance(processed_raster_bioclim, rasterio.io.DatasetReader)
    assert isinstance(processed_raster_soilgrids, rasterio.io.DatasetReader)
    assert isinstance(region_mask, rasterio.io.DatasetReader)
