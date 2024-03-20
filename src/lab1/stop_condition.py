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


class PrecisionCondition(StopCondition):
    """Condition of stop after reaching the required accuracy"""

    def __init__(self, precision) -> None:
        """
        Constructor for precision condition
        :param precision:   required accuracy
        """
        self.precision = precision

    def stop(self, point: tp.Optional[Point] = None, state: tp.Optional[State] = None) -> bool:
        if state is not None:
            return self.precision > state.eps
        return False


class AndCondition(StopCondition):
    """Condition of step when both conditions are true"""

    def __init__(self, first_condition: StopCondition, second_condition: StopCondition):
        """
        Constructor for and condition
        :param first_condition:     first condition
        :param second_condition:    second condition
        """
        self.first_condition = first_condition
        self.second_condition = second_condition

    def stop(self, point: tp.Optional[Point] = None, state: tp.Optional[State] = None) -> bool:
        return self.first_condition.stop(point, state) and self.second_condition.stop(point, state)


class OrCondition(StopCondition):
    """Condition of step when one of two conditions is true"""

    def __init__(self, first_condition: StopCondition, second_condition: StopCondition):
        """
        Constructor for or condition
        :param first_condition:     first condition
        :param second_condition:    second condition
        """
        self.first_condition = first_condition
        self.second_condition = second_condition

    def stop(self, point: tp.Optional[Point] = None, state: tp.Optional[State] = None) -> bool:
        first_result = self.first_condition.stop(point, state)
        second_result = self.second_condition.stop(point, state)
        return first_result or second_result
