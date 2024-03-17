from abc import ABC, abstractmethod
import typing as tp

from ..common import Point, State


class StopCondition(ABC):
    """Interface for stop condition of processing method"""

    @abstractmethod
    def stop(self, point: tp.Optional[Point] = None, state: tp.Optional[State] = None) -> bool:
        """
        Check need to stop
        :param point:       current point
        :param state:       current state
        :return:            is stop
        """
        pass


class CountCondition(StopCondition):
    """Condition of stop after n-count"""

    def __init__(self, max_count: int) -> None:
        """
        Constructor for count condition
        :param max_count:   count of steps before stop
        """
        self.max_count = max_count
        self.count = 0

    def stop(self, point: tp.Optional[Point] = None, state: tp.Optional[State] = None) -> bool:
        self.count += 1
        return self.count > self.max_count
