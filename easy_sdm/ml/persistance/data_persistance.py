class DataPersistance:
    def __init__(self) -> None:
        pass


# from pathlib import Path

# import pandas as pd
# import yaml
# from datamodel_code_generator import InputFileType, generate
# from pandas_profiling import ProfileReport

# from size_profile_ml.base import BaseDataPersistence


# class DataPersistence(BaseDataPersistence):
#     @classmethod
#     def __types_mapping(cls):
#         types_mapping = {
#             "object": "string",
#             "int32": "integer",
#             "int64": "integer",
#             "float32": "number",
#             "float64": "number",
#             "bool": "boolean",
#             "datetime64[ns]": "string",
#         }
#         return types_mapping

#     @classmethod
#     def __format_mapping(cls):
#         format_mapping = {
#             "object": "default",
#             "int32": "int32",
#             "int64": "int64",
#             "float32": "float",
#             "float64": "float",
#             "datetime64[ns]": "date-time",
#         }
#         return format_mapping

#     @classmethod
#     def training_dataset_profiling(
#         cls, value: pd.DataFrame, output_dir: str, **kwargs
#     ) -> pd.DataFrame:
#         profile = ProfileReport(value, **kwargs,)
#         profile.to_file(Path(output_dir) / "data_report.html")
#         profile.to_file(Path(output_dir) / "data_report.json")
#         return value

#     @classmethod
#     def features_payload_profiling(
#         cls, value: pd.DataFrame, output_dir: str, **kwargs,
#     ) -> None:
#         res = value.dtypes.to_frame("dtypes").reset_index()
#         feature_types = res.set_index("index")["dtypes"].astype(str).to_dict()

#         yaml_schema_filepath = (
#             Path(__file__).parent.parent / "config" / "payload_model_schema.yaml"
#         )
#         with open(yaml_schema_filepath) as file:
#             payload_model = yaml.full_load(file)

#         feature_list = [k for k, v in feature_types.items()]
#         payload_model["components"]["schemas"]["Payload"]["required"] = feature_list

#         payload_model["components"]["schemas"]["Payload"]["properties"] = {}
#         types_mapping = cls.__types_mapping()
#         format_mapping = cls.__format_mapping()
#         for feature_name, feature_type in feature_types.items():
#             payload_model["components"]["schemas"]["Payload"]["properties"][
#                 feature_name
#             ] = {
#                 "type": "array",
#                 "items": {
#                     "type": types_mapping.get(feature_type, "string"),
#                     "format": format_mapping.get(feature_type, "default"),
#                     "nullable": True,
#                     # TODO: this 'nullable' does nothing, it depends that all null/missing values are numpy.nans
#                     # https://github.com/koxudaxi/datamodel-code-generator/issues/437#issuecomment-870229936
#                     # https://github.com/samuelcolvin/pydantic/issues/1624
#                 },
#             }

#         output_dir = Path(output_dir)
#         yaml_filename = "payload_model.yaml"
#         yaml_filepath = output_dir / yaml_filename
#         with open(yaml_filepath, "w") as file:
#             yaml.dump(payload_model, file, sort_keys=False)
#         logger.info(f"Wrote file: '{yaml_filepath}'")

#         pyvalidator_filename = "payload_model.py"
#         output_filepath = output_dir / pyvalidator_filename
#         generate(
#             input_=Path(yaml_filepath),
#             input_file_type=InputFileType.OpenAPI,
#             output=output_filepath,
#             class_name="PayloadModel",
#         )
#         logger.info(f"Wrote file: '{output_filepath}'")
