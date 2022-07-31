import json
from pathlib import Path
from typing import Dict

import mlflow

from easy_sdm.ml.models.base import BaseEstimator


class MLFlowPersistence:
    def __init__(self, mlflow_experiment_name: str) -> None:
        self.mlflow_experiment_name = mlflow_experiment_name
        self.__setup_mlflow()
        self.__set_experiment()

    def __setup_mlflow(self):
        ml_dirpath = str(Path.cwd() / "data/ml")
        mlflow.set_tracking_uri(f"file:{ml_dirpath}")

    def __set_experiment(self):

        if not mlflow.get_experiment_by_name(self.mlflow_experiment_name):
            mlflow.create_experiment(name=self.mlflow_experiment_name)
        mlflow.set_experiment(self.mlflow_experiment_name)

    def __perist_and_log_artifacts(self, object, artifact_name: str):
        artifacts_dirpath = mlflow.get_artifact_uri().replace("file://", "")
        artifact_path = f"{artifacts_dirpath}/{artifact_name}"
        if artifact_name.endswith("json"):
            out_file = open(artifact_path, "w")
            json.dump(object, out_file)

    def __persist_logs(self, metrics: Dict, parameters: Dict):
        print(f"Logged Parameters {parameters}")
        print(f"Logged Metrics {metrics}")
        mlflow.log_params(parameters)
        mlflow.log_metrics(metrics)

    def persist(
        self,
        model: BaseEstimator,
        metrics: Dict,
        parameters: Dict,
        tags: Dict,
        end=True,
        kfold_metrics: Dict = None,
    ):

        if mlflow.active_run():
            mlflow.end_run()

        run = mlflow.start_run()
        run_id = run.info.run_id

        tags["Estimator"] = model.estimator_name
        tags["run ID"] = run_id
        mlflow.set_tags(tags=tags)

        if kfold_metrics != None:
            self.__perist_and_log_artifacts(
                object=kfold_metrics, artifact_name="kfold_metrics.json"
            )

        self.__persist_logs(metrics, parameters)

        if model.framework == "sklearn":
            mlflow.sklearn.log_model(model, self.mlflow_experiment_name)
        elif model.framework == "pytorch":
            mlflow.pytorch.log_model(model, self.mlflow_experiment_name)
        elif model.framework == "xgboost":
            mlflow.xgboost.log_model(model, self.mlflow_experiment_name)
        else:
            raise TypeError()

        if end:
            mlflow.end_run()

    def persist_and_register(self, model_name, model, metrics: Dict, parameters: Dict):
        """model_name: repository name"""
        self.persist(model, metrics, parameters, end=False)
        run_uuid = mlflow.active_run().info.run_uuid
        mlflow.register_model("runs:/{}/{}".format(run_uuid, "model"), model_name)
        mlflow.end_run()
