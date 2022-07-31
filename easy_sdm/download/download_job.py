from pathlib import Path

from enums import RasterSource

from easy_sdm.utils import PathUtils

from .sources.soilgrids_downloader import SoilgridsDownloader


class DownloadJob:
    def __init__(self, raw_rasters_dirpath) -> None:
        self.raw_rasters_dirpath = raw_rasters_dirpath
        self.__build_empty_folders()

    def __build_empty_folders(self):

        folder_list = [e.name for e in RasterSource]
        for folder in folder_list:
            PathUtils.create_folder(self.raw_rasters_dirpath / folder)

    def download_bioclim_rasters(self):
        # TODO
        raise NotImplementedError()

    def download_envirem_rasters(self):
        # TODO
        raise NotImplementedError()

    def download_shapefile_region(self):
        # TODO
        raise NotImplementedError()

    def download_soigrids_rasters(self, coverage_filter: str):
        variables = [
            "wrb",
            "cec",
            "clay",
            "phh2o",
            "silt",
            "ocs",
            "bdod",
            "cfvo",
            "nitrogen",
            "sand",
            "soc",
            "ocd",
        ]

        for variable in variables:
            self.__download_soilgrods_one_raster(variable, coverage_filter)

    def __download_soilgrods_one_raster(self, variable: str, coverage_filter: str):
        reference_raster_path = (
            Path.cwd()
            / "data"
            / "raster_processing"
            / "environment_variables_rasters"
            / RasterSource.Bioclim.name
            / "bio1_annual_mean_temperature.tif"
        )
        root_dir = self.raw_rasters_dirpath / RasterSource.Soilgrids.name
        soilgrids_downloader = SoilgridsDownloader(
            reference_raster_path=reference_raster_path, root_dir=root_dir
        )
        soilgrids_downloader.set_soilgrids_requester(variable=variable)
        coverages = soilgrids_downloader.get_coverage_list()
        for cov in coverages:
            if coverage_filter in cov:
                soilgrids_downloader.download(coverage_type=cov)
