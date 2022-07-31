class EstimatorPersistence:
    def __init__(self) -> None:
        pass


# import pickle
# from pathlib import Path

# from sklearn.base import BaseEstimator

# from size_profile_ml.base import BaseEstimatorPersistence


# class EstimatorPersistence(BaseEstimatorPersistence):
#     @classmethod
#     def load(cls, file_path: str, **kwargs) -> BaseEstimator:
#         with open(file_path, "rb") as f:
#             estimator = pickle.load(f, **kwargs)
#         return estimator

#     @classmethod
#     def dump(
#         cls,
#         estimator: BaseEstimator,
#         output_dir: str,
#         filename: str = "estimator",
#         suffix: str = "pkl",
#         **kwargs,
#     ) -> None:
#         with open(Path(output_dir) / f"{filename}.{suffix}", "wb") as output_file:
#             pickle.dump(
#                 obj=estimator, file=output_file, **kwargs,
#             )
