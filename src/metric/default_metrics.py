import numpy as np

from .metric import Metric, MetricResult
from ..common import State
from ..common.oracul import Oracul, Point, GradientOracul


def default_metrics() -> list[Metric]:
    return [CallCount(), UniqueCallCount()]


class CallCount(Metric):
    """Metric counting count of calls of oracul"""
    calls_count = 0

    def process_oracul(self, oracul: Oracul) -> Oracul:
        class CountingOracul(Oracul):
            def evaluate(self_oracul, point: Point) -> float:
                self.calls_count += 1
                return oracul.evaluate(point)

            def get_dimension(self_oracul) -> int:
                return oracul.get_dimension()

            def evaluate_gradient(self_oracul, point: Point) -> np.ndarray:
                return oracul.evaluate_gradient(point)

        return CountingOracul()

    def detect_step(self, point: Point, state: State) -> None:
        pass

    def get_result(self, method_name: str = "", **params) -> MetricResult:
        return MetricResult(
            metric_name=self.name(),
            method_name=method_name,
            result=float(self.calls_count)
        )

    def name(self) -> str:
        return "Calls"


class GradientCount(Metric):
    """Metric counting count of calls of oracul's gradient"""
    calls_count = 0

    def process_oracul(self, oracul: GradientOracul) -> Oracul:
        class CountingOracul(GradientOracul):
            def evaluate(self_oracul, point: Point) -> float:
                return oracul.evaluate(point)

            def get_dimension(self_oracul) -> int:
                return oracul.get_dimension()

            def evaluate_gradient(self_oracul, point: Point) -> np.ndarray:
                self.calls_count += 1
                return oracul.evaluate_gradient(point)

        return CountingOracul()

    def detect_step(self, point: Point, state: State) -> None:
        pass

    def get_result(self, method_name: str = "", **params) -> MetricResult:
        return MetricResult(
            metric_name=self.name(),
            method_name=method_name,
            result=float(self.calls_count)
        )

    def name(self) -> str:
        return "Gradients"


class PrecisionCount(Metric):
    """Metric counting count of calls before reaching the required accuracy"""
    step_count = 0
    stopped = False

    def __init__(self, precision) -> None:
        """Constructor of precision count metric
        :param precision:   required accuracy
        """
        self.control_precision = precision

    def process_oracul(self, oracul: Oracul) -> Oracul:
        return oracul

    def detect_step(self, point: Point, state: State) -> None:
        self.stopped |= state.eps is not None and state.eps < self.control_precision
        self.step_count += not self.stopped

    def get_result(self, method_name: str = "", **params) -> MetricResult:
        return MetricResult(
            metric_name=self.name(),
            method_name=method_name,
            result=float(self.step_count)
        )

    def name(self) -> str:
        return "Precision({})".format(self.control_precision)


class AbsolutePrecisionCount(Metric):
    """Metric counting count of calls before reaching accuracy relative absolute minimum"""
    count = 0
    stopped = False

    def __init__(self, real_point: Point, eps: float) -> None:
        self.real_point = real_point
        self.eps = eps

    def process_oracul(self, oracul: Oracul) -> Oracul:
        return oracul

    def detect_step(self, point: Point, state: State) -> None:
        self.stopped |= self.real_point.distance(point) < self.eps
        self.count += not self.stopped

    def get_result(self, method_name: str = "", **params) -> MetricResult:
        return MetricResult(
            metric_name=self.name(),
            method_name=method_name,
            result=float(self.count)
        )

    def name(self) -> str:
        return "Absolute({})".format(self.eps)


class UniqueCallCount(Metric):
    """Metric counting count of unique calls of oracul"""
    unique_calls_count = 0
    calls_from: set[tuple[Oracul, Point]] = set()

    def process_oracul(self, oracul: Oracul) -> Oracul:
        class CountingOracul(Oracul):
            def evaluate(self_oracul, point: Point) -> float:
                if (oracul, point) not in self.calls_from:
                    self.unique_calls_count += 1
                    self.calls_from.add((oracul, point))

                return oracul.evaluate(point)

            def get_dimension(self_oracul) -> int:
                return oracul.get_dimension()

            def evaluate_gradient(self_oracul, point: Point) -> np.ndarray:
                return oracul.evaluate_gradient(point)

        return CountingOracul()

    def detect_step(self, point: Point, state: State) -> None:
        pass

    def get_result(self, **params) -> float:
        return self.unique_calls_count

    def get_result(self, method_name: str = "", **params) -> MetricResult:
        return MetricResult(
            metric_name=self.name(),
            method_name=method_name,
            result=float(self.unique_calls_count)
        )

    def name(self) -> str:
        return "Unique"
