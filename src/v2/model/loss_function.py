from abc import ABC, abstractmethod

import numpy as np

from src.v2.model.oracul import EpochState


class LossFunction(ABC):
    def eval(self, data_points: np.ndarray, input_points: np.ndarray, arguments: np.ndarray, **params) -> float:
        return np.sum(np.array([self.loss(data_points[i], input_points[i], **params) for i in range(len(data_points))]))

    @abstractmethod
    def loss(self, data_point: np.ndarray, input_point: np.ndarray, **params) -> float:
        pass
