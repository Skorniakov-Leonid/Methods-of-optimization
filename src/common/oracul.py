from abc import ABC, abstractmethod
from dataclasses import dataclass
from inspect import signature
from typing import Callable

import numpy as np


@dataclass
class Point:
    """Class for n-dimensional point"""
    coordinates: np.ndarray


class Oracul(ABC):
    """Interface for oracul (anonymous value generator in point)"""

    @abstractmethod
    def evaluate(self, point: Point) -> float:
        """
        Get value in point
        :param point:       point for evaluating
        :return:            calculated value
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get dimension of oracul
        :return:            dimension
        """
        pass


class GradientOracul(Oracul):
    """Interface for oracul that can calculate gradient in point"""

    @abstractmethod
    def evaluate_gradient(self, point: Point) -> np.ndarray:
        """
        Get gradient in point
        :param point:       point for calculating gradient
        :return:            calculated gradient
        """
        pass


class LambdaOracul(Oracul):
    """Interface for oracul from lambda"""

    def __init__(self, func: Callable[..., float]) -> None:
        """
        Constructor for lambda oracul
        :param func:        lambda that generate oracul
        """
        self.func = func
        self.dimension = len(signature(func).parameters) + 1

    def evaluate(self, point: Point) -> float:
        return self.func(*(point.coordinates[:self.dimension - 1]))

    def get_dimension(self) -> int:
        return self.dimension


class PoweredSumOracul(GradientOracul):
    def __init__(self, params) -> None:
        self.params = params

    def evaluate(self, point: Point) -> float:
        res = 0
        for i in range(len(self.params)):
            res += self.params[i][0] * (point.coordinates[i] ** self.params[i][1])
        return res

    def evaluate_gradient(self, point: Point) -> np.ndarray:
        res = np.zeros(len(self.params))
        for i in range(len(self.params)):
            res[i] = (point.coordinates[i]**(self.params[i][1] - 1))*self.params[i][0]*self.params[i][1]
        return res

    def get_dimension(self) -> int:
        return len(self.params) + 1
