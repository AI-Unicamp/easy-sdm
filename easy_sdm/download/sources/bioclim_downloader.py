import requests
from pathlib import Path
# from tqdm import tqdm
import wget
import glob
import zipfile

class Unzipper:
    def unzip_all_in_folder(self, zip_file_dirpath):
        for zipfile_path in glob.glob(f"{str(zip_file_dirpath)}/*.zip"):
            self.__unzip(zipfile_path, zip_file_dirpath)

    def __unzip(self, zip_file_path, zip_file_dirpath):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(zip_file_dirpath)

class BioclimDownloader:
    def __init__(self, data_dirpath: Path) -> None:
        self.data_dirpath = data_dirpath
        self.bioclim_vars_url = "https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_30s_bio.zip"
        self.strm_elevation_url = "https://biogeo.ucdavis.edu/data/worldclim/v2.1/base/wc2.1_30s_elev.zip"
        self.unzipper = Unzipper()
        self.bioclim_dirpath = self.data_dirpath / "Bioclim"

    def download_bioclim_vars(self):
        wget.download(self.strm_elevation_url, str(self.bioclim_dirpath / "strm.zip"))
        wget.download(self.bioclim_vars_url, str(self.bioclim_dirpath / "bio.zip"))
        self.unzipper.unzip_all_in_folder(self.bioclim_dirpath)

if __name__ == "__main__":
    bioclim_downloader = BioclimDownloader(Path.cwd()/ "data_fake")
    bioclim_downloader.download_bioclim_vars()
