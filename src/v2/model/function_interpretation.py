from abc import ABC, abstractmethod

import numpy as np


class FunctionInterpretation(ABC):
    def interpretate_data(self, data_point: np.ndarray) -> np.ndarray:
        return data_point

    def evaluate(self, arguments: np.ndarray, points: np.ndarray) -> np.ndarray:
        return np.array([self.interpret(arguments, p) for p in points])

    @abstractmethod
    def interpret(self, arguments: np.ndarray, point: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        pass

    @abstractmethod
    def get_eval_dimension(self) -> int:
        pass
