import numpy as np
import typing as tp

from src.v2.model.loss_function import LossFunction


class MSE(LossFunction):
    def eval(self, data_points: np.ndarray, input_points: np.ndarray, arguments: np.ndarray, **params) -> float:
        return (np.sum(
            np.array([self.loss(data_points[i], input_points[i], **params) for i in range(len(data_points))]))
                / len(data_points))

    def loss(self, data_point: np.ndarray, input_point: np.ndarray, **params) -> float:
        return np.sum(
            np.array([(float(data_point[i]) - float(input_point[i])) ** 2 for i in range(data_point.shape[0])]))


class LineBinary(LossFunction):
    def eval(self, data_points: np.ndarray, input_points: np.ndarray, arguments: np.ndarray, **params) -> float:
        return np.sum(
            np.array([self.loss(data_points[i], input_points[i], **params) for i in range(len(data_points))]))

    def loss(self, data_point: np.ndarray, input_point: np.ndarray, **params) -> float:
        return float((1e-10
                      if ((data_point[2] == 1 and data_point[1] > input_point[1])
                          or (data_point[2] == 0 and data_point[1] < input_point[1]))
                      else (data_point[1] - input_point[1]) ** 2))


class LineBinaryWithDistance(LossFunction):
    def __init__(self, coeff: float = 1, **params) -> None:
        self.coeff = coeff

    def eval(self, data_points: np.ndarray, input_points: np.ndarray, arguments: np.ndarray, **params) -> float:
        min_distance = min([abs(data_points[index][1] - input_points[index][1]) for index in range(len(data_points))])
        return np.sum(
            np.array([self.loss(data_points[i], input_points[i], **params) for i in range(len(data_points))])) - min_distance

    def loss(self, data_point: np.ndarray, input_point: np.ndarray, **params) -> float:
        return float((1e-10
                      if ((data_point[2] == 1 and data_point[1] > input_point[1])
                          or (data_point[2] == 0 and data_point[1] < input_point[1]))
                      else (data_point[1] - input_point[1]) ** 2))


class Regularisation(LossFunction):
    def __init__(self, loss_function: LossFunction, regularisation_function: tp.Callable) -> None:
        self.loss_function = loss_function
        self.regularisation_function = regularisation_function

    def eval(self, data_points: np.ndarray, input_points: np.ndarray, arguments: np.ndarray, **params) -> float:
        return (self.loss_function.eval(data_points, input_points, arguments, **params) +
                self.regularisation_function(arguments))

    def loss(self, data_point: np.ndarray, input_point: np.ndarray, **params) -> float:
        pass


class L1(Regularisation):
    def __init__(self, loss_function: LossFunction, delta: float, **params) -> None:
        def L1_fun(arguments: np.ndarray) -> float:
            return delta * np.sum(np.absolute(arguments))

        super().__init__(loss_function, L1_fun)


class L2(Regularisation):
    def __init__(self, loss_function: LossFunction, delta: float, **params) -> None:
        def L2_fun(arguments: np.ndarray) -> float:
            return delta * np.sum(np.square(arguments))

        super().__init__(loss_function, L2_fun)

class Elastic(Regularisation):
    def __init__(self, loss_function: LossFunction, delta_L1: float, delta_L2: float, **params) -> None:
        def elastic_fun(arguments: np.ndarray) -> float:
            return delta_L1 * np.sum(np.absolute(arguments)) + delta_L2 * np.sum(np.square(arguments))

        super().__init__(loss_function, elastic_fun)
