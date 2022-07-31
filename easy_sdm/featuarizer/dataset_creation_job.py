from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split

from easy_sdm.enums import ModellingType, PseudoSpeciesGeneratorType
from easy_sdm.enums.modelling_type import ModellingType
from easy_sdm.featuarizer.dataset_builder.VIF_calculator import VIFCalculator
from easy_sdm.typos import Species
from easy_sdm.utils import PathUtils, PickleLoader
from easy_sdm.utils.data_loader import DatasetLoader

from .dataset_builder.occurrence_dataset_builder import OccurrancesDatasetBuilder
from .dataset_builder.pseudo_absense_dataset_builder import PseudoAbsensesDatasetBuilder
from .dataset_builder.scaler import MinMaxScalerWrapper


class DatasetCreationJob:
    """
    [Create a dataset with species and pseudo spescies for SDM Machine Learning]
    """

    def __init__(
        self,
        root_data_dirpath: Path,
        species: Species,
        species_gdf: gpd.GeoDataFrame,
        ps_generator_type: PseudoSpeciesGeneratorType,
        modelling_type: ModellingType,
    ) -> None:

        self.root_data_dirpath = root_data_dirpath
        self.species = species
        self.species_gdf = species_gdf
        self.ps_generator_type = ps_generator_type
        self.modelling_type = modelling_type
        self.random_state = 42
        self.k_splits = 5  # controls train-test proportion

        self.__setup()

    def __setup(self):

        self.species_dataset_path = (
            self.root_data_dirpath
            / f"featuarizer/datasets/{self.species.get_name_for_paths()}/{self.modelling_type.value}/{self.ps_generator_type.value}"
        )
        self.species_dataset_path.mkdir(parents=True, exist_ok=True)

        self.raster_path_list_path = (
            self.root_data_dirpath / "environment/relevant_raster_list"
        )

        self.region_mask_raster_path = (
            self.root_data_dirpath / "raster_processing/region_mask.tif"
        )

        self.raster_path_list = PickleLoader(self.raster_path_list_path).load_dataset()

        statistics_dataset_path = (
            self.root_data_dirpath / f"environment/raster_statistics.csv"
        )

        statistics_dataset, _ = DatasetLoader(statistics_dataset_path).load_dataset()

        self.min_max_scaler = MinMaxScalerWrapper(
            statistics_dataset=statistics_dataset,
        )

        self.occ_dataset_builder = OccurrancesDatasetBuilder(
            raster_path_list=self.raster_path_list,
        )

    def __create_occ_df(self):

        self.occ_dataset_builder.build(self.species_gdf)
        occ_df = self.occ_dataset_builder.get_dataset()
        coords_occ_df = self.occ_dataset_builder.get_coordinates_df()
        scaled_occ_df = self.min_max_scaler.scale_df(occ_df)
        return scaled_occ_df, coords_occ_df

    def __create_psa_df(
        self, scaled_train_occ_df: pd.DataFrame, number_pseudo_absenses: int
    ):

        self.psa_dataset_builder = PseudoAbsensesDatasetBuilder(
            root_data_dirpath=self.root_data_dirpath,
            ps_generator_type=self.ps_generator_type,
            scaled_occurrence_df=scaled_train_occ_df,
            min_max_scaler=self.min_max_scaler,
        )

        self.psa_dataset_builder.build(number_pseudo_absenses=number_pseudo_absenses)
        psa_decision_map = self.psa_dataset_builder.get_psa_decision_map()

        psa_df = self.psa_dataset_builder.get_dataset()
        coords_psa_df = self.psa_dataset_builder.get_coordinates_df()

        scaled_psa_df = self.min_max_scaler.scale_df(psa_df)

        return scaled_psa_df, coords_psa_df, psa_decision_map

    def __create_test_datasets(self, fold_data_dict: Dict):
        scaled_test_df = pd.concat(
            [
                fold_data_dict["scaled_test_occ_df"],
                fold_data_dict["scaled_test_psa_df"],
            ],
            ignore_index=True,
        )

        coords_test_df = pd.concat(
            [
                fold_data_dict["coords_test_occ_df"],
                fold_data_dict["coords_test_psa_df"],
            ],
            ignore_index=True,
        )

        return scaled_test_df, coords_test_df

    def __create_binary_classification_dataset(self, fold_data_dict: Dict):
        scaled_train_df = pd.concat(
            [
                fold_data_dict["scaled_train_occ_df"],
                fold_data_dict["scaled_train_psa_df"],
            ],
            ignore_index=True,
        )

        coords_train_df = pd.concat(
            [
                fold_data_dict["coords_train_occ_df"],
                fold_data_dict["coords_train_psa_df"],
            ],
            ignore_index=True,
        )

        scaled_test_df, coords_test_df = self.__create_test_datasets(
            fold_data_dict=fold_data_dict
        )

        scaled_train_df, coords_train_df = self.__shuffle_sdm_and_coords_df_sincronaly(
            sdm_df=scaled_train_df, coords_df=coords_train_df
        )
        scaled_test_df, coords_test_df = self.__shuffle_sdm_and_coords_df_sincronaly(
            sdm_df=scaled_test_df, coords_df=coords_test_df
        )

        return {
            "scaled_train_df": scaled_train_df,
            "coords_train_df": coords_train_df,
            "scaled_test_df": scaled_test_df,
            "coords_test_df": coords_test_df,
        }

    def __create_anomaly_detection_dataset(self, fold_data_dict: Dict):
        scaled_train_df = fold_data_dict["scaled_train_occ_df"].reset_index(drop=True)

        coords_train_df = fold_data_dict["coords_train_occ_df"].reset_index(drop=True)

        scaled_test_df, coords_test_df = self.__create_test_datasets(
            fold_data_dict=fold_data_dict
        )

        return {
            "scaled_train_df": scaled_train_df,
            "coords_train_df": coords_train_df,
            "scaled_test_df": scaled_test_df,
            "coords_test_df": coords_test_df,
        }

    def __shuffle_sdm_and_coords_df_sincronaly(
        self, sdm_df: pd.DataFrame, coords_df: pd.DataFrame
    ):
        sdm_df = sdm_df.sample(frac=1)
        coords_df = coords_df.iloc[list(sdm_df.index)]
        sdm_df = sdm_df.reset_index(drop=True)
        coords_df = coords_df.reset_index(drop=True)

        return sdm_df, coords_df

    def __create_full_data(
        self, scaled_occ_df: pd.DataFrame, coords_occ_df: pd.DataFrame
    ):

        if self.modelling_type == ModellingType.AnomalyDetection:
            complete_df = scaled_occ_df
            coords_df = coords_occ_df

        elif self.modelling_type == ModellingType.BinaryClassification:
            scaled_psa_df, coords_psa_df, psa_decision_map = self.__create_psa_df(
                scaled_train_occ_df=scaled_occ_df,
                number_pseudo_absenses=len(scaled_occ_df),
            )
            complete_df = pd.concat([scaled_occ_df, scaled_psa_df], ignore_index=True)
            coords_df = pd.concat([coords_occ_df, coords_psa_df], ignore_index=True)

        complete_df, coords_df = self.__shuffle_sdm_and_coords_df_sincronaly(
            sdm_df=complete_df, coords_df=coords_df
        )

        vif_calculator = self.__create_vif_calculator(data=complete_df)
        vif_df = vif_calculator.get_vif_df()
        return complete_df, coords_df, vif_df, psa_decision_map

    def __save(self, df: pd.DataFrame, dirname: str, filename: str):

        assert ".csv" in filename
        output_dirpath = self.species_dataset_path / dirname
        output_dirpath.mkdir(parents=True, exist_ok=True)

        output_path = output_dirpath / filename

        df.to_csv(output_path, index=False)

    def __save_array(self, np_array: np.ndarray, dirname: str, filename: str):

        assert ".npy" in filename
        output_dirpath = self.species_dataset_path / dirname
        output_dirpath.mkdir(parents=True, exist_ok=True)

        output_path = output_dirpath / filename

        with open(output_path, "wb") as f:
            np.save(f, np_array)

        del np_array

    def create_dataset(self,):

        kf = KFold(n_splits=self.k_splits, shuffle=True, random_state=self.random_state)
        scaled_occ_df, coords_occ_df = self.__create_occ_df()
        complete_df, coords_df, vif_df, psa_decision_map = self.__create_full_data(
            scaled_occ_df=scaled_occ_df, coords_occ_df=coords_occ_df
        )
        self.__save_array(
            np_array=psa_decision_map,
            dirname="full_data",
            filename="psa_decision_map.npy",
        )
        self.__save(df=complete_df, dirname="full_data", filename="complete_df.csv")
        self.__save(df=coords_df, dirname="full_data", filename="coords_df.csv")
        self.__save(df=vif_df, dirname="full_data", filename="vif_decision_df.csv")

        scaled_occ_df.index = scaled_occ_df.index.tolist()
        for i, (train_index, test_index) in enumerate(kf.split(scaled_occ_df)):
            kfold_number = i + 1

            scaled_train_occ_df, scaled_test_occ_df = (
                scaled_occ_df.iloc[train_index],
                scaled_occ_df.iloc[test_index],
            )
            coords_train_occ_df, coords_test_occ_df = (
                coords_occ_df.iloc[train_index],
                coords_occ_df.iloc[test_index],
            )
            scaled_psa_df, coords_psa_df, _ = self.__create_psa_df(
                scaled_train_occ_df=scaled_train_occ_df,
                number_pseudo_absenses=len(scaled_occ_df),
            )
            scaled_train_psa_df, scaled_test_psa_df = (
                scaled_psa_df.iloc[train_index],
                scaled_psa_df.iloc[test_index],
            )
            coords_train_psa_df, coords_test_psa_df = (
                coords_psa_df.iloc[train_index],
                coords_psa_df.iloc[test_index],
            )

            fold_data_dict = {}
            fold_data_dict["scaled_train_occ_df"] = scaled_train_occ_df
            fold_data_dict["scaled_test_occ_df"] = scaled_test_occ_df
            fold_data_dict["coords_train_occ_df"] = coords_train_occ_df
            fold_data_dict["coords_test_occ_df"] = coords_test_occ_df
            fold_data_dict["scaled_train_psa_df"] = scaled_train_psa_df
            fold_data_dict["scaled_test_psa_df"] = scaled_test_psa_df
            fold_data_dict["coords_train_psa_df"] = coords_train_psa_df
            fold_data_dict["coords_test_psa_df"] = coords_test_psa_df

            if self.modelling_type == ModellingType.AnomalyDetection:
                fold_dataset_dict = self.__create_anomaly_detection_dataset(
                    fold_data_dict
                )
            elif self.modelling_type == ModellingType.BinaryClassification:
                fold_dataset_dict = self.__create_binary_classification_dataset(
                    fold_data_dict
                )

            fold_vif_dict = self.__create_kfold_vif_dict(dataset_dict=fold_dataset_dict)
            self.__save_all_for_kfold(kfold_number, fold_dataset_dict, fold_vif_dict)

    def __create_vif_calculator(self, data: pd.DataFrame):
        tempdir = PathUtils.get_temp_dir()
        temp_vif_reference_df_path = tempdir / "temp.csv"
        data.to_csv(temp_vif_reference_df_path, index=False)
        vif_calculator = VIFCalculator(
            dataset_path=temp_vif_reference_df_path, output_column="label"
        )
        vif_calculator.calculate_vif()
        return vif_calculator

    def __create_kfold_vif_dict(self, dataset_dict: Dict):
        vif_calculator = self.__create_vif_calculator(
            data=dataset_dict["scaled_train_df"]
        )

        vif_dict = {}

        vif_dict["train_vif_df"] = dataset_dict["scaled_train_df"][
            vif_calculator.get_optimous_columns_with_label()
        ]
        vif_dict["test_vif_df"] = dataset_dict["scaled_test_df"][
            vif_calculator.get_optimous_columns_with_label()
        ]
        vif_dict["vif_decision_df"] = vif_calculator.get_vif_df()

        return vif_dict

    def __save_all_for_kfold(
        self, kfold_number: int, fold_datataset_dict: Dict, fold_vif_data_dict: Dict
    ):

        str_kfold_number = str(kfold_number)
        dirname = f"kfold{str_kfold_number}"

        # coords df
        self.__save(
            df=fold_datataset_dict["coords_train_df"],
            dirname=dirname,
            filename="coords_train.csv",
        )
        self.__save(
            df=fold_datataset_dict["coords_test_df"],
            dirname=dirname,
            filename="coords_test.csv",
        )

        # data df
        self.__save(
            df=fold_datataset_dict["scaled_train_df"],
            dirname=dirname,
            filename="train.csv",
        )
        self.__save(
            df=fold_datataset_dict["scaled_test_df"],
            dirname=dirname,
            filename="test.csv",
        )

        # vif df
        self.__save(
            df=fold_vif_data_dict["train_vif_df"],
            dirname=dirname,
            filename="vif_train.csv",
        )
        self.__save(
            df=fold_vif_data_dict["test_vif_df"],
            dirname=dirname,
            filename="vif_test.csv",
        )
        self.__save(
            df=fold_vif_data_dict["vif_decision_df"],
            dirname=dirname,
            filename="vif_decision_df.csv",
        )


class DatasetCreationJobPrevious:
    """
    [Create a dataset with species and pseudo spescies for SDM Machine Learning]
    """

    def __init__(
        self,
        root_data_dirpath: Path,
        species: Species,
        species_gdf: gpd.GeoDataFrame,
        ps_proportion: float,
        ps_generator_type: PseudoSpeciesGeneratorType,
        modelling_type: ModellingType,
    ) -> None:

        self.inference_proportion_from_all_data = 0.2
        self.test_proportion_from_inference_data = 0.5

        self.root_data_dirpath = root_data_dirpath
        self.species = species
        self.species_gdf = species_gdf
        self.ps_proportion = ps_proportion
        self.ps_generator_type = ps_generator_type
        self.modelling_type = modelling_type
        self.random_state = 42

        self.__setup()
        self.__build_empty_folders()

        self.train_occ = None
        self.val_occ = None
        self.test_occ = None

        self.train_psa = None
        self.val_psa = None
        self.test_psa = None

    def __build_empty_folders(self):
        PathUtils.create_folder(self.featuarizer_dirpath)

    def __setup(self):

        self.species_dataset_path = (
            self.root_data_dirpath
            / f"featuarizer/datasets/{self.species.get_name_for_paths()}/{self.modelling_type.value}"
        )
        self.species_dataset_path.mkdir(parents=True, exist_ok=True)

        self.raster_path_list_path = (
            self.root_data_dirpath / "environment/relevant_raster_list"
        )
        self.featuarizer_dirpath = self.root_data_dirpath / "featuarizer"

        self.region_mask_raster_path = (
            self.root_data_dirpath / "raster_processing/region_mask.tif"
        )

        self.raster_path_list = PickleLoader(self.raster_path_list_path).load_dataset()

        statistics_dataset = (
            self.__create_statistics_dataset()
        )  # apenas a patir do treino
        self.min_max_scaler = MinMaxScalerWrapper(
            statistics_dataset=statistics_dataset,
        )

        self.occ_dataset_builder = OccurrancesDatasetBuilder(
            raster_path_list=self.raster_path_list,
        )

    def __split_dataset(
        self, df: pd.DataFrame,
    ):

        df_train, df_ = train_test_split(
            df,
            test_size=self.inference_proportion_from_all_data,
            random_state=self.random_state,
        )
        df_valid, df_test = train_test_split(
            df_,
            test_size=self.test_proportion_from_inference_data,
            random_state=self.random_state,
        )

        return df_train, df_valid, df_test

    def __reflect_split_from_other_dataset(
        self,
        input_df: pd.DataFrame,
        sdm_df_train: pd.DataFrame,
        sdm_df_valid: pd.DataFrame,
        sdm_df_test: pd.DataFrame,
    ):
        input_df_train = input_df.iloc[list(sdm_df_train.index)]
        input_df_valid = input_df.iloc[list(sdm_df_valid.index)]
        input_df_test = input_df.iloc[list(sdm_df_test.index)]
        return input_df_train, input_df_valid, input_df_test

    def __create_statistics_dataset(self):

        # raster_statistics_calculator = DataframeStatisticsCalculator(
        #     df=df,
        # )
        # statistics_dataset_path = self.species_dataset_path / 'statistics.csv'

        statistics_dataset_path = (
            self.root_data_dirpath / f"featuarizer/raster_statistics.csv"
        )

        raster_statistics_calculator = RasterStatisticsCalculator(
            raster_path_list=self.raster_path_list,
            mask_raster_path=self.region_mask_raster_path,
        )
        raster_statistics_calculator.build_table(statistics_dataset_path)

        statistics_dataset = pd.read_csv(statistics_dataset_path)

        return statistics_dataset

    def __create_occ_df(self):

        self.occ_dataset_builder.build(self.species_gdf)
        occ_df = self.occ_dataset_builder.get_dataset()
        coords_occ_df = self.occ_dataset_builder.get_coordinates_df()

        scaled_occ_df = self.min_max_scaler.scale_df(occ_df)
        (
            scaled_train_occ_df,
            scaled_val_occ_df,
            scaled_test_occ_df,
        ) = self.__split_dataset(scaled_occ_df)

        (
            train_coords_occ_df,
            val_coords_occ_df,
            test_coords_occ_df,
        ) = self.__reflect_split_from_other_dataset(
            coords_occ_df, scaled_train_occ_df, scaled_val_occ_df, scaled_test_occ_df
        )

        self.scaled_train_occ_df = scaled_train_occ_df
        self.scaled_val_occ_df = scaled_val_occ_df
        self.scaled_test_occ_df = scaled_test_occ_df

        self.train_coords_occ_df = train_coords_occ_df
        self.val_coords_occ_df = val_coords_occ_df
        self.test_coords_occ_df = test_coords_occ_df

    def __create_psa_df(self,):
        self.psa_dataset_builder = PseudoAbsensesDatasetBuilder(
            root_data_dirpath=self.root_data_dirpath,
            ps_generator_type=self.ps_generator_type,
            scaled_occurrence_df=self.scaled_train_occ_df,
            min_max_scaler=self.min_max_scaler,
        )

        occ_df_size = (
            len(self.scaled_train_occ_df)
            + len(self.scaled_val_occ_df)
            + len(self.scaled_test_occ_df)
        )

        number_pseudo_absenses = int(occ_df_size * self.ps_proportion)
        self.psa_dataset_builder.build(number_pseudo_absenses=number_pseudo_absenses)

        psa_df = self.psa_dataset_builder.get_dataset()
        coords_psa_df = self.psa_dataset_builder.get_coordinates_df()

        scaled_psa_df = self.min_max_scaler.scale_df(psa_df)
        (
            scaled_train_psa_df,
            scaled_val_psa_df,
            scaled_test_psa_df,
        ) = self.__split_dataset(scaled_psa_df)

        (
            train_coords_psa_df,
            val_coords_psa_df,
            test_coords_psa_df,
        ) = self.__reflect_split_from_other_dataset(
            coords_psa_df, scaled_train_psa_df, scaled_val_psa_df, scaled_test_psa_df
        )

        # Estao vindo valores -9999.0
        self.scaled_train_psa_df = scaled_train_psa_df
        self.scaled_val_psa_df = scaled_val_psa_df
        self.scaled_test_psa_df = scaled_test_psa_df

        self.train_coords_psa_df = train_coords_psa_df
        self.val_coords_psa_df = val_coords_psa_df
        self.test_coords_psa_df = test_coords_psa_df

    def __create_binary_classification_dataset(self):
        self.scaled_train_df = pd.concat(
            [self.scaled_train_occ_df, self.scaled_train_psa_df], ignore_index=True
        )
        self.scaled_val_df = pd.concat(
            [self.scaled_val_occ_df, self.scaled_val_psa_df], ignore_index=True
        )
        self.scaled_test_df = pd.concat(
            [self.scaled_test_occ_df, self.scaled_test_psa_df], ignore_index=True
        )

        self.train_coords_df = pd.concat(
            [self.train_coords_occ_df, self.train_coords_psa_df], ignore_index=True
        )
        self.val_coords_df = pd.concat(
            [self.val_coords_occ_df, self.val_coords_psa_df], ignore_index=True
        )
        self.test_coords_df = pd.concat(
            [self.test_coords_occ_df, self.test_coords_psa_df], ignore_index=True
        )

    def __create_anomaly_detection_dataset(self):
        self.scaled_train_df = self.scaled_train_occ_df.reset_index(drop=True)
        self.scaled_val_df = pd.concat(
            [self.scaled_val_occ_df, self.scaled_val_psa_df], ignore_index=True
        )
        self.scaled_test_df = pd.concat(
            [self.scaled_test_occ_df, self.scaled_test_psa_df], ignore_index=True
        )

        self.train_coords_df = self.train_coords_occ_df.reset_index(drop=True)
        self.val_coords_df = pd.concat(
            [self.val_coords_occ_df, self.val_coords_psa_df], ignore_index=True
        )
        self.test_coords_df = pd.concat(
            [self.test_coords_occ_df, self.test_coords_psa_df], ignore_index=True
        )

    def create_dataset(self,):

        self.__create_occ_df()
        self.__create_psa_df()
        if self.modelling_type == ModellingType.AnomalyDetection:
            self.__create_anomaly_detection_dataset()
        elif self.modelling_type == ModellingType.BinaryClassification:
            self.__create_binary_classification_dataset()

        self.create_vif_dataset()
        self.save()

    def create_vif_dataset(self):
        tempdir = PathUtils.get_temp_dir()
        temp_vif_reference_df_path = tempdir / "temp.csv"
        self.scaled_train_df.to_csv(temp_vif_reference_df_path, index=False)
        vif_calculator = VIFCalculator(
            dataset_path=temp_vif_reference_df_path, output_column="label"
        )
        vif_calculator.calculate_vif()

        self.train_vif_df = self.scaled_train_df[
            vif_calculator.get_optimous_columns_with_label()
        ]
        self.val_vif_df = self.scaled_val_df[
            vif_calculator.get_optimous_columns_with_label()
        ]
        self.test_vif_df = self.scaled_test_df[
            vif_calculator.get_optimous_columns_with_label()
        ]
        self.vif_decision_df = vif_calculator.get_vif_df()

    def save(self):

        # coords df
        self.train_coords_df.to_csv(
            self.species_dataset_path / "train_coords.csv", index=False
        )
        self.val_coords_df.to_csv(
            self.species_dataset_path / "valid_coords.csv", index=False
        )
        self.test_coords_df.to_csv(
            self.species_dataset_path / "test_coords.csv", index=False
        )

        # data df
        self.scaled_train_df.to_csv(
            self.species_dataset_path / "train.csv", index=False
        )
        self.scaled_val_df.to_csv(self.species_dataset_path / "valid.csv", index=False)
        self.scaled_test_df.to_csv(self.species_dataset_path / "test.csv", index=False)

        # vif df
        self.train_vif_df.to_csv(
            self.species_dataset_path / "vif_train.csv", index=False
        )
        self.val_vif_df.to_csv(self.species_dataset_path / "vif_valid.csv", index=False)
        self.test_vif_df.to_csv(self.species_dataset_path / "vif_test.csv", index=False)
        self.vif_decision_df.to_csv(
            self.species_dataset_path / "vif_decision_df.csv", index=False
        )
