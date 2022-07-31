from pathlib import Path

import numpy as np

from easy_sdm.environment import EnverionmentLayersStacker


def test_env_layer_stacker(processed_raster_paths_list):
    env_lstacker = EnverionmentLayersStacker()
    stacked_raster = env_lstacker.stack_and_save(
        raster_path_list=processed_raster_paths_list
    )

    assert isinstance(stacked_raster, (np.ndarray))
