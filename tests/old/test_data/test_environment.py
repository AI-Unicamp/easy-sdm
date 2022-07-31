from pathlib import Path
from statistics import variance
from tkinter import Variable

import numpy as np
import rasterio

from easy_sdm.configs import configs
from easy_sdm.raster_processing import (
    RasterCliper,
    RasterLoader,
    RasterStandarizer,
    SoilgridsDownloader,
)
from easy_sdm.utils import PathUtils


def test_download_soilgrids(tmp_path, mock_processed_raster_path):
    variable = "clay"
    soilgrids_downloader = SoilgridsDownloader(
        reference_raster_path=mock_processed_raster_path, root_dir=tmp_path
    )
    soilgrids_downloader.set_soilgrids_requester(variable=variable)
    coverage_example = soilgrids_downloader.get_coverage_list()[3]
    soilgrids_downloader.download(coverage_type=coverage_example)
    # coverage_example = coverage_example.replace('.','_')
    output_path = Path(tmp_path / variable / f"{coverage_example}.tif")
    assert output_path.is_file()
    assert isinstance(
        RasterLoader(str(output_path)).load_dataset(), rasterio.io.DatasetReader
    )


def test_clip_raster(tmp_path, raw_rasters_dirpath):

    filepath = PathUtils.get_rasters_filepaths_in_dir(raw_rasters_dirpath)[0]
    filename = PathUtils.get_rasters_filenames_in_dir(raw_rasters_dirpath)[0]
    output_path = tmp_path / filename
    raster = RasterLoader(filepath).load_dataset()
    RasterCliper().clip_and_save(
        source_raster=raster, output_path=PathUtils.file_path_existis(output_path),
    )
    assert Path(output_path).is_file()
    assert isinstance(
        RasterLoader(output_path).load_dataset(), rasterio.io.DatasetReader
    )


def test_species_inside_regions(tmp_path, mock_map_shapefile_path):
    import geopandas as gpd

    from easy_sdm.raster_processing import (
        Species,
        SpeciesGDFBuilder,
        SpeciesInShapefileChecker,
    )

    mays_code = 5290052
    mays = SpeciesGDFBuilder(Species(taxon_key=mays_code, name="Zea mays"))
    mays_gdf = mays.get_species_gdf()
    shp_region = SpeciesInShapefileChecker(mock_map_shapefile_path)
    new_mays_gdf = shp_region.get_points_inside(mays_gdf)
    mays.save_species_gdf
    assert type(new_mays_gdf) is gpd.GeoDataFrame
    assert len(new_mays_gdf) <= len(mays_gdf)


def test_standarize_soilgrids_raster(tmp_path):
    """[Este teste esta dando errado
    Uma possivel solucao seria jogar todos os pontos 0.0 para -9999.0 e em seguida filtrar
    pelo mapa do brasil e jogar todos os que estivessem dentro para 0. Precisa pensar se
    isso faz sentido.
    ]
    """

    raster_path = "data/raw/rasters/Soilgrids_Rasters/bdod/bdod_15-30cm_mean.tif"
    output_path = tmp_path / Path(raster_path).name
    raster_standarizer = RasterStandarizer()
    raster_standarizer.standarize_soilgrids(
        input_path=raster_path, output_path=output_path,
    )
    standarized_raster = rasterio.open(output_path)

    assert np.min(standarized_raster.read(1)) is configs["maps"]["no_data_val"]
