import pytest

from easy_sdm.download import DownloadJob


@pytest.mark.interface
def test_download_job(tmp_path):
    # TODO:create monkey patches
    download_job = DownloadJob(raw_rasters_dirpath=tmp_path)
    download_job.download_shapefile_region()
    download_job.download_soigrids_rasters(coverage_filter="mean")
    download_job.download_bioclim_rasters()
    download_job.download_envirem_rasters()
