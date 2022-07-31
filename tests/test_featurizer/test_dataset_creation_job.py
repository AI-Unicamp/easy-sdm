import pytest

from easy_sdm.enums import ModellingType, PseudoSpeciesGeneratorType
from easy_sdm.featuarizer import DatasetCreationJob
from easy_sdm.utils.data_loader import DatasetLoader, ShapefileLoader


@pytest.fixture
def dataset_creation_job(root_test_data_path):

    return DatasetCreationJob(root_data_dirpath=root_test_data_path,)


def test_create_binary_classification_dataset(
    mock_species,
    mock_species_shapefile_path,
    dataset_creation_job,
    mock_featuarizer_dirpath,
):

    scaled_df, coords_df = dataset_creation_job.create_binary_classification_dataset(
        species_gdf=ShapefileLoader(mock_species_shapefile_path).load_dataset(),
        ps_generator_type=PseudoSpeciesGeneratorType.RSEP,
        ps_proportion=0.5,
    )

    dataset_creation_job.save_dataset(
        species=mock_species,
        sdm_df=scaled_df,
        coords_df=coords_df,
        modellting_type=ModellingType.BinaryClassification,
    )
    binary_classification_folder = (
        mock_featuarizer_dirpath / "datasets/cajanus_cajan/binary_classification"
    )

    assert (
        DatasetLoader(
            dataset_path=binary_classification_folder / "train.csv",
            output_column="label",
        )
        .load_dataset()[0]
        .shape[1]
        == DatasetLoader(
            dataset_path=binary_classification_folder / "train_coords.csv",
            output_column="",
        )
        .load_dataset()[0]
        .shape[1]
    )
