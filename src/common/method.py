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
    :param eps              current maximum deviation
    """
    visual_state: list[Figure]
    parameters: list[tp.Any]
    eps: tp.Optional[float] = None


class OptimizationMethod(ABC):
    """Interface for optimization method"""

    @abstractmethod
    def initial_step(self, oracul: Oracul, visualize: bool = False, **params) -> tuple[Point, State]:
        """
        Initial step of method
        :param oracul:      anonymous value generator in point
        :param params:      optional parameters
        :param visualize:   need visualize
        :return:            founded point and state for next step
        """
        pass

    @abstractmethod
    def step(self, oracul: Oracul, state: State, visualize: bool = False) -> tuple[Point, State]:
        """
        Step of method
        :param oracul:      anonymous value generator in point
        :param state:       state for step
        :param visualize:   need visualize
        :return:            founded point and state for next step
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """
        Name of method
        :return:            name of method
        """
        pass
