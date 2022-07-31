from pathlib import Path

from easy_sdm.environment import EnvironmentCreationJob
from easy_sdm.utils import NumpyArrayLoader, PickleLoader


def test_environment_creation_job(
    processed_raster_paths_list, mock_environment_dirpath
):

    env_creation_job = EnvironmentCreationJob(
        output_dirpath=mock_environment_dirpath,
        all_rasters_path_list=processed_raster_paths_list,
    )
    env_creation_job.build_environment()
    environment_stack = NumpyArrayLoader(
        dataset_path=mock_environment_dirpath / "environment_stack.npy"
    ).load_dataset()
    relevant_raster_list = PickleLoader(
        dataset_path=mock_environment_dirpath / "relevant_raster_list"
    ).load_dataset()
    assert len(environment_stack.shape) == 3
    assert Path.exists(relevant_raster_list[0])
