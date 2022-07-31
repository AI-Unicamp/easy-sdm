import json
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from easy_sdm.configs import configs
from easy_sdm.eda import EDAJob
from easy_sdm.utils.data_loader import RasterLoader
from easy_sdm.utils.path_utils import PathUtils

app = typer.Typer()


milpa_species_dict = {
    5290052: "Zea mays",
    7393329: "Cucurbita moschata",
    2874515: "Cucurbita maxima",
    2874508: "Cucurbita pepo",
    5350452: "Phaseolus vulgaris",
    2982583: "Vigna unguiculata",
    7587087: "Cajanus cajan",
    3086357: "Piper nigrum",
    2932944: "Capsicum annuum",
    2932938: "Capsicum baccatum",
    8403992: "Capsicum frutescens",
    2932942: "Capsicum chinense",
}
milpa_species_dict = {
    5290052: "Zea mays",
    2874508: "Cucurbita pepo",
    7587087: "Cajanus cajan",
    2932944: "Capiscum annuum",
}

data_dirpath = Path.cwd() / "data"


def version_callback(value: bool):
    if value:
        with open(Path(__file__).parent / "VERSION", mode="r") as file:
            version = file.read().replace("\n", "")
        typer.echo(f"{version}")
        raise typer.Exit()


@app.callback("version")
def version(
    version: Optional[bool] = typer.Option(
        None, "--version", callback=version_callback, is_eager=True,
    ),
):
    """
    Any issue please contact authors
    """
    typer.echo("easy_sdm")


@app.command("check-processed-rasters")
def check_processed_rasters():
    """
    Em todos os mrus rasters existem resquicios de -9999.0 dentro da mascara. O que comprometera
    o algoritimo futuramente. Preciso ter certeza de que todos eles foram removidos antes da proxima
    etapa.
    """
    raster_paths = PathUtils.get_rasters_filepaths_in_dir(
        data_dirpath / "raster_processing/environment_variables_rasters"
    )
    region_mask_array = (
        RasterLoader(data_dirpath / "raster_processing/region_mask.tif")
        .load_dataset()
        .read(1)
    )
    for path in raster_paths:
        print(path)
        if "bio1_annual_mean_temperature" in str(path):
            raster = RasterLoader(path).load_dataset()
            raster_array = raster.read(1)
            raster_array = np.where(
                region_mask_array == configs["maps"]["no_data_val"],
                -1000,
                raster_array,
            )
            if raster_array.min() == configs["maps"]["no_data_val"]:
                vals, counts = np.unique(raster_array, return_counts=True)
                print([(v, c) for (v, c) in sorted(zip(counts, vals))][::-1][:5])
                import pdb

                pdb.set_trace()
            else:
                import pdb

                pdb.set_trace()


def split_dataset(df):
    from sklearn.model_selection import train_test_split

    random_state = 1
    y = df["label"]
    X = df.drop(["label"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


@app.command("models-comparision")
def models_comparision():
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.neural_network import MLPClassifier

    from easy_sdm.ml.models.tabnet import TabNetProxy

    df_result = pd.DataFrame(
        columns=[
            "species_name",
            "mlp",
            "mlp_simplified",
            "gradient_boosting",
            "gradient_boosting_simplified",
        ]
    )
    datasets_dirpath = Path.cwd() / "data/featuarizer"
    vif_dirpath = Path.cwd() / "data/output/vif_analysis"
    for id, name in milpa_species_dict.items():
        species_name = name.replace(" ", "_")

        # import pdb;pdb.set_trace()
        # species dataframe
        species_dataset_path = datasets_dirpath / f"dataset_{species_name}.csv"
        species_df = pd.read_csv(species_dataset_path)

        # vif dataframe
        species_vif_path = vif_dirpath / f"vif_{species_name}.csv"
        vif_df = pd.read_csv(species_vif_path)

        # simplified dataframe
        simplified_species_df = species_df[vif_df["feature"].to_list() + ["label"]]
        X_train, X_test, y_train, y_test = split_dataset(species_df)

        # split dataset
        (
            X_train_simplified,
            X_test_simplified,
            y_train_simplified,
            y_test_simplified,
        ) = split_dataset(simplified_species_df)

        # creating classifiers
        clf_mlp = MLPClassifier(
            hidden_layer_sizes=(200, 100, 50, 20, 10), random_state=1, max_iter=8000
        ).fit(X_train, y_train)

        clf_mlp_simplified = MLPClassifier(
            hidden_layer_sizes=(200, 100, 50, 20, 10), random_state=1, max_iter=8000
        ).fit(X_train_simplified, y_train_simplified)

        clf_gradient_boosting = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=1,
            criterion="squared_error",
        ).fit(X_train, y_train)

        clf_gradient_boosting_simplified = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=1,
            criterion="squared_error",
        ).fit(X_train_simplified, y_train_simplified)

        # len_train = len(X_train)
        # clf_tabnet = TabNetProxy(device_name='cpu').fit(
        #         X_train[:int(0.8*len_train)], y_train[:int(0.8*len_train)],
        #         X_train[int(0.8*len_train):], y_train[int(0.8*len_train):],
        #         patience = 50,
        #         max_epochs=100,
        #         batch_size = 8,
        #         virtual_batch_size = 2
        #     )

        # clf_tabnet_simplified = TabNetProxy(device_name='cpu').fit(
        #         X_train[:int(0.8*len_train)], y_train[:int(0.8*len_train)],
        #         X_train[int(0.8*len_train):], y_train[int(0.8*len_train):],
        #         patience = 50,
        #         max_epochs=100,
        #         batch_size = 8,
        #         virtual_batch_size = 2
        #     )

        # creating scores
        mlp_result = accuracy_score(y_true=y_test, y_pred=clf_mlp.predict(X_test))
        mlp_result_simplified = accuracy_score(
            y_true=y_test_simplified,
            y_pred=clf_mlp_simplified.predict(X_test_simplified),
        )

        graient_boosting_result = accuracy_score(
            y_true=y_test, y_pred=clf_gradient_boosting.predict(X_test)
        )
        graient_boosting_result_simplified = accuracy_score(
            y_true=y_test_simplified,
            y_pred=clf_gradient_boosting_simplified.predict(X_test_simplified),
        )

        df_result = df_result.append(
            {
                "species_name": species_name,
                "mlp": mlp_result,
                "mlp_simplified": mlp_result_simplified,
                "gradient_boosting": graient_boosting_result,
                "gradient_boosting_simplified": graient_boosting_result_simplified,
            },
            ignore_index=True,
        )
        # tabnet_result = accuracy_score(y_true=y_test, y_pred=clf_tabnet.predict(X_test))
        # tabnet_result_simplified = accuracy_score(
        #     y_true=y_test_simplified,
        #     y_pred=clf_tabnet_simplified.predict(X_test_simplified),
        # )


if __name__ == "__main__":
    app()
