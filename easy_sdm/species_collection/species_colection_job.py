from pathlib import Path

from easy_sdm.typos import Species
from easy_sdm.utils import PathUtils, ShapefileLoader

from .species_data.species_gdf_builder import SpeciesGDFBuilder
from .species_data.species_in_shapefile_checker import SpeciesInShapefileChecker


class SpeciesCollectionJob:
    def __init__(self, output_dirpath: Path, region_shapefile_path: Path) -> None:

        self.output_dirpath = output_dirpath
        self.shp_region = ShapefileLoader(region_shapefile_path).load_dataset()
        self.species_in_shp_checker = SpeciesInShapefileChecker(region_shapefile_path)

    def __build_empty_folders(self, dirpath):
        PathUtils.create_folder(dirpath)

    def collect_species_data(
        self, species: Species,
    ):
        species_gdf_builder = SpeciesGDFBuilder(species, self.shp_region)
        species_name = species.get_name_for_paths()
        dirpath = self.output_dirpath / species_name
        self.__build_empty_folders(dirpath)
        file_path = dirpath / f"{species_name}.shp"
        species_gdf_builder.save_species_gdf(file_path)
