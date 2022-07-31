class ExperimentPersistence:
    def __init__(self) -> None:
        pass


# import json
# from pathlib import Path
# from typing import Dict

# import pandas as pd
# from numpyencoder import NumpyEncoder

# from size_profile_ml.base import BaseExperimentPersistence


# class ExperimentPersistence(BaseExperimentPersistence):
#     @classmethod
#     def save_scores(
#         cls, metrics: Dict[str, float], output_dir: str, file_name: str
#     ) -> None:
#         destination = Path(output_dir) / f"{file_name}.json"
#         json.encoder.FLOAT_REPR = lambda o: format(o, ".2f")
#         with open(destination, "w") as f:
#             json.dump(metrics, f, sort_keys=True, indent=2, cls=NumpyEncoder)

#     @classmethod
#     def save_predictions(
#         cls,
#         predictions: pd.DataFrame,
#         output_dir: str,
#         file_name: str,
#         suffix: str = "csv",
#     ) -> None:
#         destination = Path(output_dir) / f"{file_name}.{suffix}"
#         predictions.to_csv(
#             destination,
#             index=False,
#             header=True,
#         )
