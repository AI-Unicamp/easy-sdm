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
EDA_JOB = EDAJob(data_dirpath)


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


@app.command("count-spcies-occurrences")
def count_spcies_occurrences():

    species_occurances_dict = EDA_JOB.count_occurances_per_species()
    json.dump(
        species_occurances_dict, open(data_dirpath / "eda/spcies_occurances.json", "w")
    )


@app.command("verify-processed-rasters")
def verify_processed_rasters():
    dict_varialbes = EDA_JOB.verify_processed_rasters()
    json.dump(
        dict_varialbes, open(data_dirpath / "eda/value_counts_per_raster.json", "w")
    )


@app.command("save-plots-processed-rasters")
def save_plots_processed_rasters():
    EDA_JOB.save_plots_processed_rasters()


@app.command("save-points-in-blank-map")
def save_plots_processed_rasters():
    EDA_JOB.save_plots_points_in_blank_map(milpa_species_dict)


@app.command("save-ocsvm-decision-map")
def save_ocsvm_decision_map():
    EDA_JOB.save_ocsvm_decision_map(milpa_species_dict)


if __name__ == "__main__":
    app()
