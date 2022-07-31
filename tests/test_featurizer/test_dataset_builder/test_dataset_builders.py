import pytest

from easy_sdm.enums import PseudoSpeciesGeneratorType
from easy_sdm.featuarizer import OccurrancesDatasetBuilder, PseudoAbsensesDatasetBuilder
from easy_sdm.utils import ShapefileLoader


@pytest.fixture
def occurance_dataset_builder(mock_species_shapefile_path, processed_raster_paths_list):
    occ_dataset_builder = OccurrancesDatasetBuilder(processed_raster_paths_list)
    occ_dataset_builder.build(
        ShapefileLoader(mock_species_shapefile_path).load_dataset()
    )
    return occ_dataset_builder


@pytest.fixture
def pseudo_absense_dataset_builder(
    occurance_dataset_builder, mock_mask_raster_path, mock_environment_dirpath
):

    psa_dataset_builder = PseudoAbsensesDatasetBuilder(
        ps_generator_type=PseudoSpeciesGeneratorType.RSEP,
        region_mask_raster_path=mock_mask_raster_path,
        stacked_raster_coverages_path=mock_environment_dirpath
        / "environment_stack.npy",
    )

    occ_df = occurance_dataset_builder.get_dataset()
    psa_dataset_builder.build(occurrence_df=occ_df, number_pseudo_absenses=len(occ_df))

    return psa_dataset_builder


def test_extract_occurances(occurance_dataset_builder):

    occ_df = occurance_dataset_builder.get_dataset()
    coordinates_df = occurance_dataset_builder.get_coordinates_df()
    assert occ_df.index.equals(coordinates_df.index)


def test_extract_pseudo_absenses(pseudo_absense_dataset_builder):

    psa_df = pseudo_absense_dataset_builder.get_dataset()
    coordinates_df = pseudo_absense_dataset_builder.get_coordinates_df()

    assert psa_df.index.equals(coordinates_df.index)
