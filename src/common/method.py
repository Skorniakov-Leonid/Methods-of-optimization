from abc import ABC, abstractmethod
from dataclasses import dataclass
import typing as tp

from .figure import Figure
from .oracul import Point, Oracul


@dataclass
class State:
    """Class for state transferring between steps
    :param visual_state     list of geometric figure that visualize current state
    :param parameters       parameters for next step
    """
    visual_state: list[Figure]
    parameters: list[tp.Any]


class OptimizationMethod(ABC):
    """Interface for optimization method"""

    @abstractmethod
    def initial_step(self, oracul: Oracul, **params) -> tuple[Point, State]:
        """
        Initial step of method
        :param oracul:      anonymous value generator in point
        :param params:      optional parameters
        :return:            founded point and state for next step
        """
        pass

    @abstractmethod
    def step(self, oracul: Oracul, state: State) -> tuple[Point, State]:
        """
        Step of method
        :param oracul:      anonymous value generator in point
        :param state:       state for step
        :return:            founded point and state for next step
        """
        pass
