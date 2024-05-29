from inspect import signature
from math import sin
from typing import Callable

import numpy as np

from src.v2.model.function_interpretation import FunctionInterpretation


class LinearInterpretation(FunctionInterpretation):
    def interpret(self, arguments: np.ndarray, point: np.ndarray) -> np.ndarray:
        return np.array([point[0], arguments[0] * point[0] + arguments[1]])

    def get_dimension(self) -> int:
        return 3

    def get_eval_dimension(self) -> int:
        return 2


class MultiLinearInterpretation(FunctionInterpretation):
    def __init__(self, dim):
        self.dim = dim

    def interpret(self, arguments: np.ndarray, point: np.ndarray) -> np.ndarray:
        res = np.array(point)
        res = np.append(res, np.dot(arguments[0:self.dim], point) + arguments[-1])
        return res

    def get_dimension(self) -> int:
        return self.dim + 1

    def get_eval_dimension(self) -> int:
        return self.dim


class PolynomialInterpretation(FunctionInterpretation):
    def __init__(self, argument_count: int = 3, **params) -> None:
        self.argument_count = argument_count

    def interpret(self, arguments: np.ndarray, point: np.ndarray) -> np.ndarray:
        x = point[0]
        return np.array([x, sum([arg * x ** i for i, arg in enumerate(arguments[:self.argument_count])])])

    def get_dimension(self) -> int:
        return self.argument_count + 1

    def get_eval_dimension(self) -> int:
        return 2


class HyperbolicInterpretation(FunctionInterpretation):
    def interpret(self, arguments: np.ndarray, point: np.ndarray) -> np.ndarray:
        return np.array([point[0],
                         arguments[0] * point[0] ** 3 + arguments[1] * point[0] ** 2 + arguments[2] * point[0] +
                         arguments[3]])

    def get_dimension(self) -> int:
        return 4

    def get_eval_dimension(self) -> int:
        return 2


class MatrixInterpretation(FunctionInterpretation):
    def __init__(self, dim, eval_dim):
        self.dim = dim
        self.eval_dim = eval_dim

    def interpretate_data(self, data_point: np.ndarray) -> np.ndarray:
        vector = [0] * self.eval_dim
        vector[int(data_point[-1])] = 1
        return np.array(vector)

    def interpret(self, arguments: np.ndarray, point: np.ndarray) -> np.ndarray:
        classes = np.dot(arguments.reshape(self.eval_dim, self.dim), point.transpose())
        return classes / np.linalg.norm(classes, ord=1)

    def get_dimension(self) -> int:  # num of parameters
        return self.dim

    def get_eval_dimension(self) -> int:  # num of classes
        return self.eval_dim


class LambdaInterpretation(FunctionInterpretation):
    """
    func(a1, a2, a3 ... an, point)
    """

    def __init__(self, func: Callable[..., np.ndarray], eval_dim: int = 2) -> None:
        self.func = func
        self.dimension = len(signature(func).parameters)
        self.eval_dim = eval_dim

    def interpret(self, arguments: np.ndarray, point: np.ndarray) -> np.ndarray:
        return self.func(*arguments, point)

    def get_dimension(self) -> int:
        return self.dimension

    def get_eval_dimension(self) -> int:
        return self.eval_dim
