import numpy as np

from src.v2.model.loss_function import LossFunction


class MSE(LossFunction):
    def eval(self, data_points: np.ndarray, input_points: np.ndarray, **params) -> float:
        return (np.sum(
            np.array([self.loss(data_points[i], input_points[i], **params) for i in range(len(data_points))]))
                / len(data_points))

    def loss(self, data_point: np.ndarray, input_point: np.ndarray, **params) -> float:
        return np.sum(np.array([(float(data_point[i]) - float(input_point[i])) ** 2 for i in range(data_point.shape[0])]))
