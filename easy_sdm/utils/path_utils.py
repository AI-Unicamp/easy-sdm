import os
import shutil
import tempfile
import time
from pathlib import Path


class TemporaryDirectory:
    def __init__(self) -> None:
        self.name = Path(tempfile.mkdtemp())

    def free(self):
        shutil.rmtree(self.name)


class PathUtils:
    def __init__(self):
        pass

    @classmethod
    def get_temp_dir(cls):
        tempdir = Path(tempfile.mkdtemp())
        return tempdir

    @classmethod
    def create_folder(cls, dirpath: Path):
        p = Path(dirpath)
        p.mkdir(parents=True, exist_ok=True)
        while not dirpath.is_dir():
            time.sleep(1)

    @classmethod
    def file_path(cls, string):
        if Path(string).is_file():
            return Path(string)
        else:
            raise FileNotFoundError(string)

    @classmethod
    def file_path_existis(cls, string):
        if Path(string).is_file():
            raise FileExistsError()
        else:
            return Path(string)

    @classmethod
    def dir_path(cls, string):
        if Path(string).is_dir():
            return Path(string)
        else:
            raise NotADirectoryError(string)

    @classmethod
    def get_rasters_filepaths_in_dir(cls, dirpath: Path):
        list_filepaths = []

        for root, _, files in os.walk(dirpath):  # root ,dirs, files

            for file in files:
                if file.endswith(".tif") and "mask" not in file:
                    list_filepaths.append(Path(root) / file)
        return list_filepaths

    @classmethod
    def get_rasters_filenames_in_dir(cls, dirpath: Path):
        list_filenames = []

        for _, _, files in os.walk(dirpath):  # root ,dirs, files

            for file in files:
                if file.endswith(".tif") and "mask" not in file:
                    list_filenames.append(file)
        return list_filenames

    @classmethod
    def get_rasters_varnames_in_dir(cls, dirpath: Path):
        list_varnames = []

        for _, _, files in os.walk(dirpath):  # root ,dirs, files

            for file in files:
                if file.endswith(".tif") and "mask" not in file:
                    name = file.replace(".tif", "")
                    list_varnames.append(name)
        return list_varnames
