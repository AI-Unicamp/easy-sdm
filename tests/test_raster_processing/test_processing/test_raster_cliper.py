from pathlib import Path

import rasterio

from easy_sdm.raster_processing import RasterCliper
from easy_sdm.utils import PathUtils, RasterLoader


def test_clip_raster(tmp_path, mock_processed_raster_path_bioclim):

    output_path = tmp_path / "clipped_raster.tif"
    raster = RasterLoader(mock_processed_raster_path_bioclim).load_dataset()
    RasterCliper().clip_and_save(
        source_raster=raster, output_path=PathUtils.file_path_existis(output_path),
    )
    loaded_raster = RasterLoader(output_path).load_dataset()
    assert Path(output_path).is_file()
    assert isinstance(loaded_raster, rasterio.io.DatasetReader)
    # No Brasil a distancia W é um pouco maior que a distancia L
    assert loaded_raster.shape[1] > loaded_raster.shape[0]
