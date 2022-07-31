from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


class MinMaxScalerWrapper:
    """ "
    Mix max scale dataframe and enverionment stack
    Columns should be passed because they will keed the enverionment mssvariables ordoer
    """

    def __init__(self, statistics_dataset: pd.DataFrame) -> None:
        self.statistics_dataset = statistics_dataset.set_index("raster_name")

    def __scale(self, data: np.ndarray):
        # epsilon=1e-100
        min_vec = self.statistics_dataset["min"].to_numpy()
        max_vec = self.statistics_dataset["max"].to_numpy()
        data_scaled = (data - min_vec) / (max_vec - min_vec)
        return data_scaled

    def scale_df(self, df: pd.DataFrame):
        label = df["label"].to_numpy()
        df = df.drop("label", axis=1)
        index = df.index
        columns = df.columns
        values = df.to_numpy()
        values_scaled = self.__scale(values)
        scaled_df = pd.DataFrame(values_scaled, columns=columns)
        scaled_df["label"] = label
        scaled_df.index = index
        return scaled_df

    def scale_coverages(self, coverages: np.ndarray):
        scaled_coverages = self.__scale(coverages)
        return scaled_coverages

    def scale_stack(self, stack):
        stack = stack.transpose(1, 2, 0)
        stack_scaled = self.__scale(stack)
        stack_scaled = stack_scaled.transpose(2, 0, 1)
        return stack_scaled
