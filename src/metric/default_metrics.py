import numpy as np

from .metric import Metric
from ..common import State
from ..common.oracul import Oracul, Point


def default_metrics() -> list[Metric]:
    return [CallCount(), UniqueCallCount()]


class CallCount(Metric):
    """Metric counting count of calls of oracul"""
    calls_count = 0

    def process_oracul(self, oracul: Oracul) -> Oracul:
        class CountingOracul(Oracul):
            def evaluate(self_oracul, point: Point) -> np.floating:
                self.calls_count += 1
                return oracul.evaluate(point)

            def get_dimension(self_oracul) -> int:
                return oracul.get_dimension()

            def evaluate_gradient(self_oracul, point: Point):
                return oracul.evaluate_gradient(point)

        return CountingOracul()

    def detect_step(self, point: Point, state: State) -> None:
        pass

    def get_result(self, **params) -> float:
        return float(self.calls_count)


class GradientCount(Metric):
    """Metric counting count of calls of oracul's gradient"""
    calls_count = 0

    def process_oracul(self, oracul: Oracul) -> Oracul:
        class CountingOracul(Oracul):
            def evaluate(self_oracul, point: Point) -> np.floating:
                return oracul.evaluate(point)

            def get_dimension(self_oracul) -> int:
                return oracul.get_dimension()

            def evaluate_gradient(self_oracul, point: Point):
                self.calls_count += 1
                return oracul.evaluate_gradient(point)

        return CountingOracul()

    def detect_step(self, point: Point, state: State) -> None:
        pass

    def get_result(self, **params) -> float:
        return float(self.calls_count)


class StepCountBeforePrecision(Metric):
    step_count = 0

    def __init__(self, precision):
        self.control_precision = precision
        self.stop_flag = False

    def process_oracul(self, oracul: Oracul) -> Oracul:
        class CountingOracul(Oracul):
            def evaluate(self_oracul, point: Point) -> np.floating:
                return oracul.evaluate(point)

            def get_dimension(self_oracul) -> int:
                return oracul.get_dimension()

            def evaluate_gradient(self_oracul, point: Point):
                return oracul.evaluate_gradient(point)

        return CountingOracul()

    def detect_step(self, point: Point, state: State) -> None:
        if state.eps > self.control_precision and not self.stop_flag:
            self.step_count += 1
        else:
            self.stop_flag = True

    def get_result(self, **params) -> float:
        return float(self.step_count)


class UniqueCallCount(Metric):
    """Metric counting count of unique calls of oracul"""
    unique_calls_count = 0
    calls_from: set[tuple[Oracul, Point]] = set()

    def process_oracul(self, oracul: Oracul) -> Oracul:
        class CountingOracul(Oracul):
            def evaluate(self_oracul, point: Point) -> np.floating:
                if (oracul, point) not in self.calls_from:
                    self.unique_calls_count += 1
                    self.calls_from.add((oracul, point))

                return oracul.evaluate(point)

            def get_dimension(self_oracul) -> int:
                return oracul.get_dimension()

        return CountingOracul()

    def detect_step(self, point: Point, state: State) -> None:
        pass

    def get_result(self, **params) -> float:
        return self.unique_calls_count
