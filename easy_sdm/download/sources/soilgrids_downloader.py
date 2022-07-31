from pathlib import Path

from owslib.wcs import WebCoverageService

from easy_sdm.configs import configs
from easy_sdm.utils import PathUtils, RasterLoader, RasterUtils


class SoilgridsDownloader:
    """[summary]

    References:
    - The soilgrids lib:
    - The possible soilgrids variables: https://maps.isric.org/
    """

    def __init__(self, reference_raster_path: Path, root_dir: Path) -> None:

        self.reference_raster = RasterLoader(
            raster_path=reference_raster_path
        ).load_dataset()
        self.root_dir = root_dir
        self.soilgrids_requester = None
        self.variable = None
        self.configs = configs
        self.width = self.reference_raster.width
        self.height = self.reference_raster.height

    def set_soilgrids_requester(self, variable: str):
        url = f"http://maps.isric.org/mapserv?map=/map/{variable}.map"
        self.soilgrids_requester = WebCoverageService(url, version="1.0.0")
        self.variable = variable

    def __check_if_soilgrids_requester(self):
        if self.soilgrids_requester is None:
            raise ValueError("Please set soilgrids_requester_before")

    def __build_bbox(self):
        # check bounding box
        limits = self.configs["maps"]["region_limits_with_security"]
        if limits["west"] > limits["east"] or limits["south"] > limits["north"]:
            raise ValueError(
                "Please provide valid bounding box values for west, east, south and north."
            )
        else:
            bbox = (limits["west"], limits["south"], limits["east"], limits["north"])
        return bbox

    def download(
        self, coverage_type: str,
    ):
        self.__check_if_soilgrids_requester()
        output_dir = self.root_dir / self.variable
        PathUtils.create_folder(output_dir)
        coverage_type = coverage_type.replace(".", "_")
        output_path = output_dir / f"{coverage_type}.tif"
        if not Path(output_path).is_file():
            crs = "urn:ogc:def:crs:EPSG::4326"
            response = None
            i = 0
            while response is None and i < 5:
                try:
                    response = self.soilgrids_requester.getCoverage(
                        identifier=coverage_type,
                        crs=crs,
                        bbox=self.__build_bbox(),
                        resx=None,
                        resy=None,
                        width=self.width,
                        height=self.height,
                        response_crs=crs,
                        format="GEOTIFF_INT16",
                    )  # bytes
                    RasterUtils.save_binary_raster(response, output_path)
                except Exception as e:
                    print(type(e))

            i += 1
        else:
            print(f"{coverage_type} already downloaded")

    def get_coverage_list(self):
        self.__check_if_soilgrids_requester()
        return list(self.soilgrids_requester.contents)
