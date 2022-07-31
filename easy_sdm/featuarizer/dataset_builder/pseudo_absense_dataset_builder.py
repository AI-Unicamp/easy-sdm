from pathlib import Path

import pandas as pd

from easy_sdm.enums import PseudoSpeciesGeneratorType
from easy_sdm.featuarizer.dataset_builder.scaler import MinMaxScalerWrapper
from easy_sdm.utils import NumpyArrayLoader
from easy_sdm.utils.data_loader import RasterLoader

from .pseudo_species_generators import RSEPPseudoSpeciesGenerator


class PseudoAbsensesDatasetBuilder:
    def __init__(
        self,
        root_data_dirpath: Path,
        ps_generator_type: PseudoSpeciesGeneratorType,
        scaled_occurrence_df: pd.DataFrame,
        min_max_scaler: MinMaxScalerWrapper,
    ):

        self.root_data_dirpath = root_data_dirpath
        self.ps_generator_type = ps_generator_type
        self.scaled_occurrence_df = scaled_occurrence_df
        self.min_max_scaler = min_max_scaler

        self.__setup()
        self.__define_ps_generator()

    def __setup(self):

        self.region_mask_raster_path = (
            self.root_data_dirpath / "raster_processing/region_mask.tif"
        )

        self.stacked_raster_coverages_path = (
            self.root_data_dirpath / "environment/environment_stack.npy"
        )

    def __get_var_names_list(self):
        return [Path(path).name.split(".")[0] for path in self.raster_path_list]

    def __define_ps_generator(self):
        region_mask_raster = RasterLoader(self.region_mask_raster_path).load_dataset()

        if self.ps_generator_type is PseudoSpeciesGeneratorType.RSEP:
            stacked_raster_coverages = NumpyArrayLoader(
                self.stacked_raster_coverages_path
            ).load_dataset()
            ps_generator = RSEPPseudoSpeciesGenerator(
                region_mask_raster=region_mask_raster,
                stacked_raster_coverages=stacked_raster_coverages,
                min_max_scaler=self.min_max_scaler,
            )
            ps_generator.fit(self.scaled_occurrence_df)

        elif self.ps_generator_type is PseudoSpeciesGeneratorType.Random:
            raise NotImplementedError()

        else:
            raise ValueError()

        self.ps_generator = ps_generator

    def build(self, number_pseudo_absenses: int):

        scaled_pseudo_absenses_df, coordinates_df = self.ps_generator.generate(
            number_pseudo_absenses
        )
        scaled_pseudo_absenses_df["label"] = 0
        self.dataset = scaled_pseudo_absenses_df
        self.coordinates_df = coordinates_df

    def get_dataset(self):
        return self.dataset

    def get_coordinates_df(self):
        return self.coordinates_df

    def get_psa_decision_map(self):
        return self.ps_generator.get_psa_decision_map()
