from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class Point:
    """Class for n-dimensional point"""
    coordinates: np.ndarray


class Oracul(ABC):
    """Interface for oracul (anonymous value generator in point)"""

    @abstractmethod
    def evaluate(self, point: Point) -> np.dtype[float].type:
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

    def __init__(self, func: Callable[..., float], dimension: int) -> None:
        """
        Constructor for lambda oracul
        :param func:        lambda that generate oracul
        :param dimension:   dimension where work lambda
        """
        self.func = func
        self.dimension = dimension

    def evaluate(self, point: Point) -> np.dtype[float].type:
        return self.func(*(point.coordinates[:self.dimension - 1]))

    def get_dimension(self) -> int:
        return self.dimension
