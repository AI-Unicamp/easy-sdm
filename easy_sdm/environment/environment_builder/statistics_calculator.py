from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from easy_sdm.configs import configs


class DataframeStatisticsCalculator:
    def __init__(self, df: pd.DataFrame,) -> None:
        df = df.drop("label", axis=1)
        self.df = df

    def build_table(self, output_path: Path):
        stats_df = pd.DataFrame(
            columns=["raster_name", "min", "max", "mean", "std", "median"]
        )
        for col in self.df.columns:
            column_info = self.df[col]

            stats_df = stats_df.append(
                {
                    "raster_name": col,
                    "min": np.min(column_info),
                    "max": np.max(column_info),
                    "mean": np.mean(column_info),
                    "std": np.std(column_info),
                    "median": np.median(column_info),
                },
                ignore_index=True,
            )

        stats_df.to_csv(output_path, index=False)


class RasterStatisticsCalculator:
    """[A class to extract basic statistics from rasters considering only the masked territorry]

    Attention: There is a problem in worldclim variables. They are not beeing well filtered in the south
    """

    def __init__(self, raster_path_list, mask_raster_path) -> None:
        self.configs = configs
        self.raster_path_list = raster_path_list
        self.mask_raster = rasterio.open(mask_raster_path)
        self.inside_mask_idx = np.where(
            self.mask_raster.read(1) != configs["maps"]["no_data_val"]
        )

    def build_table(self, output_path: Path):
        df = pd.DataFrame(
            columns=["raster_name", "min", "max", "mean", "std", "median"]
        )
        for raster_path in self.raster_path_list:
            raster = rasterio.open(raster_path)
            raster_data = raster.read(1)
            inside_mask_vec = raster_data[
                self.inside_mask_idx[0], self.inside_mask_idx[1]
            ]

            filtered_vec = inside_mask_vec[
                inside_mask_vec != configs["maps"]["no_data_val"]
            ]

            df = df.append(
                {
                    "raster_name": Path(raster_path).name.split(".")[0],
                    "min": np.min(filtered_vec),
                    "max": np.max(filtered_vec),
                    "mean": np.mean(filtered_vec),
                    "std": np.std(filtered_vec),
                    "median": np.median(filtered_vec),
                },
                ignore_index=True,
            )

        df.to_csv(output_path, index=False)
