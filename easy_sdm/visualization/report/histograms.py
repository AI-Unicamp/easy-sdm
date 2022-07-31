import os

import matplotlib.pyplot as plt
import pandas as pd
import rasterio


class HistogramsPlotter:
    def __init__(self) -> None:
        pass

    def saving_kfold_train_env_var_histograms(
        species_name,
        species_kfold_preditions_folder,
        utils_methods,
        output_folder,
        base_txt_files_path,
    ):
        kfold_dirs = [
            os.path.join(species_kfold_preditions_folder, name)
            for name in os.listdir(species_kfold_preditions_folder)
            if os.path.isdir(os.path.join(species_kfold_preditions_folder, name))
        ]
        for i, fold in enumerate(kfold_dirs):
            env_data_train = utils_methods.retrieve_data_from_np_array(
                os.path.join(fold, "Species_Raster_Data_Train.npy")
            )  # (N,38)
            list_names_raster = (
                open(f"{base_txt_files_path}/list_names_raster.txt", "r")
                .read()
                .splitlines()
            )
            env_data_train_df = pd.DataFrame(env_data_train, columns=list_names_raster)
            env_data_train_df.hist(layout=(10, 4), figsize=(20, 20))
            plt.suptitle(
                f"Variáveis ambientais não escaladas para as espécie {species_name} kfold{i+1}",
                fontsize=20,
            )
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            plt.savefig(f"{output_folder}/train_input_vars_unscaled_kfold{i+1}.png")
            plt.show()
            plt.clf()
