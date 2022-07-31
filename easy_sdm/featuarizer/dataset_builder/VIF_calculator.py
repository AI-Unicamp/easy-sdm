from cmath import inf
from pathlib import Path

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from easy_sdm.utils import DatasetLoader


class VIFCalculator:
    def __init__(self, dataset_path: Path, output_column: str) -> None:
        self.X = None
        self.dataset_path = dataset_path
        self.output_column = output_column

    def __calculate_vif_dataframe(self, X):
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns

        # calculating VIF for each feature
        vif_data["VIF"] = [
            variance_inflation_factor(X.values, i) for i in range(len(X.columns))
        ]
        return vif_data

    def calculate_vif(self,):
        dataloader = DatasetLoader(
            dataset_path=self.dataset_path, output_column=self.output_column
        )
        X, self.y = dataloader.load_dataset()
        max = inf
        while max > 10:
            # VIF dataframe
            vif_data = self.__calculate_vif_dataframe(X)
            idmax = vif_data["VIF"].idxmax()
            max = vif_data["VIF"].max()
            feature_to_remove = vif_data.iloc[idmax]["feature"]
            X = X.drop([feature_to_remove], axis=1)
        vif_data = self.__calculate_vif_dataframe(X)

        self.X = X
        self.vif_data = vif_data

    def get_optimous_columns(self):
        assert self.X is not None
        return self.X.columns.to_list()

    def get_optimous_columns_with_label(self):
        assert self.X is not None
        return self.get_optimous_columns() + [self.output_column]

    def get_vif_df(self):
        assert self.vif_data is not None
        return self.vif_data

    def get_optimouns_df(self):
        assert self.X is not None
        df = self.X.copy()
        df[self.output_column] = self.y.to_numpy()
        return self.X


# if __name__ == "__main__":
#     datataset_loader = DatasetLoader(dataset_path=Path.cwd() / "data/featuarizer/dataset.csv",
#               output_column=  "label")

#     X, y = datataset_loader.load_dataset()
#     vif_calculator = VIFCalculator()
#     vif_calculator.calculate_vif(X)
