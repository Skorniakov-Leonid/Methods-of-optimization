from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numdifftools as nd

from src.v2.model.meta import Meta


@dataclass
class OraculMeta(Meta):
    """Class container for meta information about oracul"""
    pass


class Oracul(ABC):
    """Interface for oracul (anonymous value generator in point)"""

    @abstractmethod
    def evaluate(self, point: np.ndarray, **params) -> float:
        """
        Get value in point
        :param point:       point for evaluating
        :return:            calculated value
        """
        pass

    def evaluate_gradient(self, point: np.ndarray, **params) -> np.ndarray:
        """
        Get gradient in point
        :param point:       point for calculating gradient
        :return:            calculated gradient
        """
        return nd.Gradient(self.evaluate)(point)

    def evaluate_hessian(self, point: np.ndarray, **params) -> np.ndarray:
        """
        Get hessian in point
        :param point:       point for calculating gradient
        :return:            calculated hessian
        """
        return nd.Hessian(self.evaluate)(point)

    @abstractmethod
    def get_dimension(self) -> int:
        pass

    @abstractmethod
    def meta(self, **params) -> OraculMeta:
        pass
