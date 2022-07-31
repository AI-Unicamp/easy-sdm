from pathlib import Path

import rasterio

from easy_sdm.download import SoilgridsDownloader
from easy_sdm.utils import RasterLoader


def test_download_soilgrids(tmp_path, mock_processed_raster_path_bioclim):
    # TODO:create monkey patches
    variable = "clay"
    soilgrids_downloader = SoilgridsDownloader(
        reference_raster_path=mock_processed_raster_path_bioclim, root_dir=tmp_path
    )
    soilgrids_downloader.set_soilgrids_requester(variable=variable)
    coverage_example = soilgrids_downloader.get_coverage_list()[3]
    soilgrids_downloader.download(coverage_type=coverage_example)
    output_path = Path(tmp_path / variable / f"{coverage_example}.tif")
    assert output_path.is_file()
    assert isinstance(
        RasterLoader(str(output_path)).load_dataset(), rasterio.io.DatasetReader
    )
