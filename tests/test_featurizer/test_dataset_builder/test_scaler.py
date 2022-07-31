from pathlib import Path
from typing import List

from easy_sdm.environment import EnverionmentLayersStacker
from easy_sdm.featuarizer import MinMaxScalerWrapper


def test_scale_dataset(processed_raster_paths_list: List):
    num_vars = 5
    processed_raster_paths_list = processed_raster_paths_list[:num_vars]
    MinMaxScalerWrapper(raster_path_list=processed_raster_paths_list)


def test_scale_enverioment(processed_raster_paths_list):
    num_vars = len(processed_raster_paths_list)
    processed_raster_paths_list = processed_raster_paths_list[:num_vars]
    stack = EnverionmentLayersStacker(processed_raster_paths_list).stack()
    scaler_wraper = MinMaxScalerWrapper(raster_path_list=processed_raster_paths_list)
    scaled_stack = scaler_wraper.scale_stack(stack=stack, statistics_dataset=df_stats)
    import pdb

    pdb.set_trace()
    assert scaled_stack.shape[0] == num_vars
