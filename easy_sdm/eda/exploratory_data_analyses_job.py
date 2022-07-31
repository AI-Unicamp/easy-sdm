from pathlib import Path

import numpy as np
from configs import configs

from easy_sdm.typos import Species
from easy_sdm.utils import RasterLoader, ShapefileLoader
from easy_sdm.utils.path_utils import PathUtils
from easy_sdm.visualization.debug_plots import (
    EnvironmentVariablesMapPlotter,
    MapWithCoords,
    MapWithOCSVMDecision,
)


class EDAJob:
    def __init__(self, data_dirpath: Path) -> None:
        self.data_dirpath = data_dirpath

    def count_occurances_per_species(self):
        dict_species = {}
        species_dirpath = self.data_dirpath / "species_collection"
        for path in species_dirpath.glob("**/*.shp"):
            species_name = path.name.replace(".shp", "")
            species_gdf = ShapefileLoader(path).load_dataset()
            dict_species[species_name] = len(species_gdf)

        return dict_species

    def save_ocsvm_decision_map(self, milpa_species_dict):
        for species_id, species_name in milpa_species_dict.items():
            species = Species(taxon_key=species_id, name=species_name)
            map_with_coords = MapWithOCSVMDecision(
                data_dirpath=self.data_dirpath, species=species
            )
            map_with_coords.plot_map()

    def save_plots_points_in_blank_map(self, milpa_species_dict):
        for species_id, species_name in milpa_species_dict.items():
            species = Species(taxon_key=species_id, name=species_name)
            map_with_coords = MapWithCoords(
                data_dirpath=self.data_dirpath, species=species
            )
            map_with_coords.plot_map()

    def save_plots_processed_rasters(self):
        raster_plotter = EnvironmentVariablesMapPlotter(self.data_dirpath)
        processed_rasters_dirpath = self.data_dirpath / "raster_processing"
        for raster_path in PathUtils.get_rasters_filepaths_in_dir(
            processed_rasters_dirpath
        ):
            print(raster_path.name)
            raster = RasterLoader(raster_path=raster_path).load_dataset()
            raster_array = raster.read(1)
            raster_plotter.plot_map(Z=raster_array, variable_name=raster_path.name)

    def verify_processed_rasters(self):

        problematic_rasters = [
            "bio14_precipitation_of_driest_month.tif",
            "bio19_precipitation_of_coldest_quarter.tif",
            "bio20_strm_worldclim_elevation.tif",
            "envir2_aridity_index_thornthwaite.tif",
            "envir3_climatic_moisture_index.tif",
            "envir4_continentality.tif",
            "envir17_envirem_terrain_roughness_index.tif",
        ]
        processed_rasters_dirpath = self.data_dirpath / "raster_processing"
        region_mask_array = (
            RasterLoader(processed_rasters_dirpath / "region_mask.tif")
            .load_dataset()
            .read(1)
        )
        dict_varialbes = {
            raster_path.name: {}
            for raster_path in PathUtils.get_rasters_filepaths_in_dir(
                processed_rasters_dirpath
            )
        }
        for raster_path in PathUtils.get_rasters_filepaths_in_dir(
            processed_rasters_dirpath
        ):
            env_var_name = raster_path.name
            if env_var_name in problematic_rasters:
                raster = RasterLoader(raster_path=raster_path).load_dataset()
                raster_array = raster.read(1)
                raster_array = np.where(
                    region_mask_array == configs["maps"]["no_data_val"],
                    "not_valid",
                    raster_array,
                )
                values, counts = np.unique(raster_array, return_counts=True)
                # import pdb;pdb.set_trace()
                for val, count in zip(values, counts):
                    dict_varialbes[env_var_name][str(val)] = str(count)

                import json

                json.dump(
                    dict_varialbes,
                    open(self.data_dirpath / "eda/value_counts_per_raster.json", "a"),
                )

        return dict_varialbes
