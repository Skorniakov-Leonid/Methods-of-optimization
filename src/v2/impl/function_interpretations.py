import numpy as np

from src.v2.model.function_interpretation import FunctionInterpretation


class LinearInterpretation(FunctionInterpretation):
    def interpret(self, arguments: np.ndarray, point: np.ndarray) -> np.ndarray:
        return np.array([point[0], arguments[0] * point[0] + arguments[1]])

    def get_dimension(self) -> int:
        return 3

    def get_eval_dimension(self) -> int:
        return 2


class PolynomialInterpretation(FunctionInterpretation):
    def interpret(self, arguments: np.ndarray, point: np.ndarray) -> np.ndarray:
        return np.array([point[0], arguments[0] * point[0] ** 2 + arguments[1] * point[0] + arguments[2]])

    def get_dimension(self) -> int:
        return 4

    def get_eval_dimension(self) -> int:
        return 2
