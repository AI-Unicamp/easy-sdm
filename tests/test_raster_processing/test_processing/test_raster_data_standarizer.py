import numpy as np

from easy_sdm.raster_processing.processing.raster_data_standarizer import (
    RasterMissingValueFiller,
)


def test_fill_missing_values():
    missing_vals = [0, -9999]
    input_arr = np.array(
        [[0.0, 4, 0, 7.0, 2.0], [3.0, 0, -9999.0, 8.0, 0], [4.0, 9.0, 6.0, 6, 0]]
    )
    expected_output = np.array(
        [
            [4.0, 4.0, 4.0, 7.0, 2.0],
            [3.0, 4.0, 8.0, 8.0, 2.0],
            [4.0, 9.0, 6.0, 6.0, 6.0],
        ]
    )
    filler = RasterMissingValueFiller()
    output_arr = filler.fill_missing_values(data=input_arr, no_value_list=missing_vals)
    assert (output_arr == expected_output).all()
