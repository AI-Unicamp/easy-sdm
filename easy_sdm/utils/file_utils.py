import json
import os

import pandas as pd


class FileUtils(object):

    """
    Class with utilities for reading and writing
    """

    def __init__(self):
        raise NotImplementedError(
            "This method is here only for checking. Should not be used"
        )

    def read_json(self, filepath: str):
        with open(filepath) as f:
            json_result = json.load(f)
        return json_result

    def reads_json(self, filepath: str):
        with open(filepath) as f:
            json_result = json.loads(f.read())
        json_result = json.loads(json_result)
        return json_result

    def dump_json(self, filepath: str, d, ensure_ascii=False, command="a"):
        with open(filepath, command) as fp:
            json.dump(d, fp, ensure_ascii=ensure_ascii)

    def dumps_json(self, filepath: str, d, ensure_ascii=False, command="a"):
        with open(filepath, command) as fp:
            d = json.dumps(d, ensure_ascii=ensure_ascii)
            json.dump(d, fp, ensure_ascii=ensure_ascii)

    def read_pickle(self, filepath: str):
        with open(filepath, "rb") as f:
            content = read_pickle(f)
        return content

    def save_pickle(self, filepath: str, info):
        """
        Save info in a picke file
        """
        with open(filepath, "wb") as f:
            dump_pickle(info, f)

    def create_folder_structure(self, folder: str):
        """Create the comple folder structure if it does not exists"""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def read_line_spaced_txt_file(self, filepath: str):
        with open(filepath, "r") as infile:
            data = infile.read().splitlines()
        return data

    def save_line_spaced_txt_file(self, filepath: str, text_list: List[str]):
        with open(filepath, "w") as output:
            for row in text_list:
                output.write(str(row) + "\n")

    def save_df_to_csv(self, filepath: str, df: pd.DataFrame, zipped=False):
        if filepath.split(".")[-1] != "csv":
            raise ValueError(f"{filepath} tem de ter a extensão .csv")

        filepath = filepath.split(".csv")[0] + ".csv.gz" if zipped else filepath
        compression = "gzip" if zipped else "infer"
        df.to_csv(filepath, compression=compression, index=False)

    def read_csv_to_df(self, filepath: str):
        if ".csv" not in filepath and ".csv.gz" not in filepath:
            raise ValueError(f"{filepath} tem de ter a extensão .csv ou csv.gz")
        df = pd.read_csv(filepath)
        return df

    def retrieve_data_from_np_array(self, path):
        """Read a numpy array"""
        with open(path, "rb") as f:
            np_array = np.load(f)
        return np_array

    def create_folder_structure(self, folder):
        """Create the comple folder structure if it does not exists"""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def save_nparray_to_folder(self, np_array, folder_path, filename):
        """Save numpy array to the specified folder path"""
        complete_path = os.path.join(folder_path, filename + ".npy")
        with open(complete_path, "wb") as f:
            print(f"{filename} Shape: ", np_array.shape)
            np.save(f, np_array)
