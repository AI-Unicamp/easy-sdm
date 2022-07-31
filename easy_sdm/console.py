from pathlib import Path
from typing import Optional

import typer

from easy_sdm.download import DownloadJob
from easy_sdm.enums import ModellingType, PseudoSpeciesGeneratorType
from easy_sdm.enums.estimator_type import EstimatorType
from easy_sdm.environment import EnvironmentCreationJob
from easy_sdm.featuarizer import DatasetCreationJob
from easy_sdm.ml import KfoldTrainJob, SimpleTrainJob
from easy_sdm.ml.prediction_job import Prediction_Job
from easy_sdm.raster_processing import RasterProcessingJob
from easy_sdm.species_collection import SpeciesCollectionJob
from easy_sdm.typos import Species
from easy_sdm.utils import PathUtils
from easy_sdm.utils.data_loader import DatasetLoader, ShapefileLoader

app = typer.Typer()

all_milpa_species_dict = {
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
all_algorithims_string = "mlp, gradient_boosting, ensemble_forest, xgboost, xgboostrf, tabnet, ocsvm, autoencoder"
# estimator selection
def estimator_type_selector(estimator_type: str):
    estimator_type = estimator_type.lower()
    estimator_type = {
        "mlp": EstimatorType.MLP,
        "gradient_boosting": EstimatorType.GradientBoosting,
        "random_forest": EstimatorType.RandomForest,
        "xgboost": EstimatorType.Xgboost,
        "xgboostrf": EstimatorType.XgboostRF,
        "tabnet": EstimatorType.Tabnet,
        "ocsvm": EstimatorType.OCSVM,
        "autoencoder": EstimatorType.Autoencoder,
    }.get(estimator_type, f"{estimator_type}' is not supported!")
    return estimator_type


# modellling Type slection from estimator
def modellling_type_selector_from_estimator(estimator_type: str):
    modelling_type = {
        EstimatorType.MLP: ModellingType.BinaryClassification,
        EstimatorType.GradientBoosting: ModellingType.BinaryClassification,
        EstimatorType.RandomForest: ModellingType.BinaryClassification,
        EstimatorType.Xgboost: ModellingType.BinaryClassification,
        EstimatorType.XgboostRF: ModellingType.BinaryClassification,
        EstimatorType.Tabnet: ModellingType.BinaryClassification,
        EstimatorType.OCSVM: ModellingType.AnomalyDetection,
        EstimatorType.Autoencoder: ModellingType.AnomalyDetection,
    }.get(estimator_type, None)
    return modelling_type


def modellling_type_selector(modelling_type: str):
    modelling_type = modelling_type.lower()
    modelling_type = {
        "binary_classification": ModellingType.BinaryClassification,
        "anomaly_detection": ModellingType.AnomalyDetection,
    }.get(modelling_type, None)
    return modelling_type


def ps_generator_type_selector(ps_generator_type):
    ps_generator_type = ps_generator_type.lower()
    ps_generator_type = {
        "rsep": PseudoSpeciesGeneratorType.RSEP,
        "random": PseudoSpeciesGeneratorType.Random,
    }.get(ps_generator_type, f"{ps_generator_type}' is not supported!")

    return ps_generator_type


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


@app.command("download-data")
def download_data():
    raw_rasters_dirpath = data_dirpath / "download/raw_rasters"
    download_job = DownloadJob(raw_rasters_dirpath=raw_rasters_dirpath)
    # download_job.download_shapefile_region()
    download_job.download_soigrids_rasters(coverage_filter="mean")
    # download_job.download_bioclim_rasters()
    # download_job.download_envirem_rasters()


@app.command("process-rasters")
def process_rasters():

    raster_processing_job = RasterProcessingJob(data_dirpath=data_dirpath)
    raster_processing_job.process_rasters_from_all_sources()


@app.command("create-all-species-data")
def create_all_species_data():
    for specie_id, species_name in milpa_species_dict.items():
        create_species_data(species_id=specie_id)


@app.command("create-species-data")
def create_species_data(species_id: int = typer.Option(..., "--species-id", "-s"),):
    output_dirpath = data_dirpath / "species_collection"
    region_shapefile_path = data_dirpath / "download/region_shapefile"
    species = Species(taxon_key=species_id, name=milpa_species_dict[species_id])
    job = SpeciesCollectionJob(
        output_dirpath=output_dirpath, region_shapefile_path=region_shapefile_path
    )
    job.collect_species_data(species=species)


@app.command("create-environment")
def create_environment():

    processed_rasters_dir = (
        data_dirpath / "raster_processing/environment_variables_rasters"
    )

    # tomar muito cuidado com essa lista porque a ordem fica baguncada
    # o stacker tem que manter a mesma ordem dos dataframes
    all_rasters_path_list = PathUtils.get_rasters_filepaths_in_dir(
        processed_rasters_dir
    )

    env_creation_job = EnvironmentCreationJob(
        data_dirpath=data_dirpath, all_rasters_path_list=all_rasters_path_list
    )
    env_creation_job.build_environment()


def create_dataset_by_specie(
    species_id: int, ps_generator_type: str, modelling_type: ModellingType,
):
    species = Species(taxon_key=species_id, name=milpa_species_dict[species_id])

    ps_generator_type = ps_generator_type_selector(ps_generator_type)

    species_gdf = ShapefileLoader(
        shapefile_path=data_dirpath
        / "species_collection"
        / species.get_name_for_paths()
    ).load_dataset()

    sdm_dataset_creator = DatasetCreationJob(
        root_data_dirpath=data_dirpath,
        species=species,
        species_gdf=species_gdf,
        ps_generator_type=ps_generator_type,
        modelling_type=modelling_type,
    )

    sdm_dataset_creator.create_dataset()


@app.command("create-dataset")
def create_dataset(
    species_id: int = typer.Option(..., "--species-id", "-s"),
    modelling_type: str = typer.Option(..., "--modelling_type", "-m"),
    ps_generator_type: str = typer.Option(..., "--ps-generator-type", "-t"),
):
    modelling_type = modellling_type_selector(modelling_type)

    create_dataset_by_specie(
        species_id=species_id,
        ps_generator_type=ps_generator_type,
        modelling_type=modelling_type,
    )


@app.command("create-all-species-datasets")
def create_dataset(
    ps_generator_type: str = typer.Option(..., "--ps-generator-type", "-t"),
):

    for species_id, species_name in milpa_species_dict.items():
        print(f"Creating dataset for: {species_name}")
        for modelling_type in [
            ModellingType.BinaryClassification,
            # ModellingType.AnomalyDetection,
        ]:
            create_dataset_by_specie(
                species_id=species_id,
                ps_generator_type=ps_generator_type,
                modelling_type=modelling_type,
            )


@app.command("train-all-estimators-all-species")
def train_all_estimators_all_species():
    for specie_id, species_name in milpa_species_dict.items():
        train_all_estimators(species_id=specie_id)


@app.command("train-all-estimators")
def train_all_estimators(species_id: int = typer.Option(..., "--species-id", "-s")):

    working_estimators = [
        "mlp",
        "gradient_boosting",
        "random_forest",
        "xgboost",
    ]
    for ps_generator_type in [PseudoSpeciesGeneratorType.RSEP.value]:
        for estimator_type in working_estimators:
            train(
                species_id=species_id,
                estimator_type=estimator_type,
                ps_generator_type=ps_generator_type,
            )


@app.command("train")
def train(
    species_id: int = typer.Option(..., "--species-id", "-s"),
    estimator_type: str = typer.Option(
        ..., "--estimator", "-e", help=f"Must be one between {all_algorithims_string}"
    ),
    ps_generator_type: str = typer.Option(..., "--ps-generator-type", "-t"),
):

    estimator_type = estimator_type_selector(estimator_type)
    modelling_type = modellling_type_selector_from_estimator(estimator_type)
    ps_generator_type = ps_generator_type_selector(ps_generator_type)
    # useful info
    species = Species(taxon_key=species_id, name=milpa_species_dict[species_id])

    # train job
    train_job = KfoldTrainJob(
        data_dirpath=data_dirpath,
        estimator_type=estimator_type,
        modelling_type=modelling_type,
        ps_generator_type=ps_generator_type,
        species=species,
    )

    train_job.run_experiment(only_vif_columns=False)
    train_job.run_experiment(only_vif_columns=True)


@app.command("simple-train")
def simple_train(
    species_id: int = typer.Option(..., "--species-id", "-s"),
    estimator_type: str = typer.Option(
        ..., "--estimator", "-e", help=f"Must be one between {all_algorithims_string}"
    ),
    ps_generator_type: str = typer.Option(..., "--ps-generator-type", "-t"),
):

    estimator_type = estimator_type_selector(estimator_type)
    modelling_type = modellling_type_selector_from_estimator(estimator_type)
    ps_generator_type = ps_generator_type_selector(ps_generator_type)
    # useful info
    species = Species(taxon_key=species_id, name=milpa_species_dict[species_id])
    dataset_dirpath = (
        data_dirpath
        / f"featuarizer/datasets/{species.get_name_for_paths()}/{modelling_type.value}"
    )
    # dataloaders
    train_data_loader = DatasetLoader(
        dataset_path=dataset_dirpath / "train.csv", output_column="label"
    )
    validation_data_loader = DatasetLoader(
        dataset_path=dataset_dirpath / "valid.csv", output_column="label"
    )
    vif_train_data_loader = DatasetLoader(
        dataset_path=dataset_dirpath / "vif_train.csv", output_column="label"
    )
    vif_validation_data_loader = DatasetLoader(
        dataset_path=dataset_dirpath / "vif_valid.csv", output_column="label"
    )

    # train job
    train_job = SimpleTrainJob(
        train_data_loader=train_data_loader,
        validation_data_loader=validation_data_loader,
        estimator_type=estimator_type,
        species=species,
    )

    train_job.fit()
    train_job.persist()

    train_job.vif_setup(
        vif_train_data_loader=vif_train_data_loader,
        vif_validation_data_loader=vif_validation_data_loader,
    )

    train_job.fit()
    train_job.persist()


@app.command("infer-map")
def infer_map(
    species_id: int = typer.Option(..., "--species-id", "-s"),
    run_id: str = typer.Option(..., "--run_id", "-r"),
):
    # TODO: selecionar o numero de features do vif. Vai precisar saber qual o numero da coluna que vai precisar filtrar
    species = Species(taxon_key=species_id, name=milpa_species_dict[species_id])
    prediction_job = Prediction_Job(
        data_dirpath=data_dirpath, run_id=run_id, species=species
    )
    prediction_job.set_model()
    Z = prediction_job.map_prediction()
    # prediction_job.log_map_with_coords(Z=Z)
    prediction_job.log_map_without_coords(Z=Z)


@app.command("infer-all-maps-for-specie")
def infer_all_maps_for_species(
    species_id: int = typer.Option(..., "--species-id", "-s"),
):
    import mlflow
    from mlflow.entities import ViewType
    from mlflow.tracking.client import MlflowClient

    ml_dirpath = str(Path.cwd() / "data/ml")
    mlflow.set_tracking_uri(f"file:{ml_dirpath}")
    species = Species(taxon_key=species_id, name=milpa_species_dict[species_id])
    experiment_id = [
        exp.experiment_id
        for exp in MlflowClient().list_experiments()
        if exp.name == species.get_name_for_plots()
    ]
    runs = MlflowClient().search_runs(
        experiment_ids=experiment_id, run_view_type=ViewType.ALL
    )
    for i in range(len(runs)):
        run_id = runs[i].data.tags["run ID"]
        infer_map(species_id=species.taxon_key, run_id=run_id)


@app.command("infer-all-maps-all-species")
def infer_all_maps_all_species():
    for specie_id, species_name in milpa_species_dict.items():
        infer_all_maps_for_species(species_id=specie_id)


if __name__ == "__main__":
    app()
