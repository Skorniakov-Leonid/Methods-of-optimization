from abc import abstractmethod, ABC
from dataclasses import dataclass

import numpy as np

from src.v2.model.meta import Meta
from src.v2.model.oracul import Oracul, EpochState


@dataclass
class State:
    """Class for state transferring between steps"""
    point: np.ndarray
    eps: float
    index: int = 1
    epoch_state: EpochState = EpochState()


@dataclass
class MethodMeta(Meta):
    """Class container for meta information about optimization method"""
    pass


class OptimizationMethod(ABC):
    """Interface for optimization method"""

    @abstractmethod
    def initial_step(self, oracul: Oracul, point: np.ndarray, **params) -> State:
        """
        Initial step of method
        :param oracul:      anonymous value generator in point
        :param point:       start point
        :param params:      optional parameters
        :return:            founded point and state for next step
        """
        pass

    @abstractmethod
    def step(self, oracul: Oracul, state: State, **params) -> State:
        """
        Step of method
        :param oracul:      anonymous value generator in point
        :param state:       state for step
        :param params:      optional parameters
        :return:            founded point and state for next step
        """
        pass

    @abstractmethod
    def meta(self, **params) -> MethodMeta:
        """
        Get meta information of method
        :return:            meta information
        """
        pass
