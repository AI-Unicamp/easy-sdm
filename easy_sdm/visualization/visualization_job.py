import numpy as np


class VisualizationJob:
    def __init__(self) -> None:
        pass

    def __build_empty_folders(self):
        raise NotImplementedError()

    def generatete_map(self, Z: np.ndarray):
        raise NotImplementedError()
