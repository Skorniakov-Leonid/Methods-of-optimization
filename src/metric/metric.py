from abc import ABC, abstractmethod

from ..common import OptimizationMethod, Oracul, Point, State


class Metric(ABC):
    """Interface for metric"""

    def process_oracul(self, oracul: Oracul) -> Oracul:
        """
        Prepare oracul to collecting metrics
        :param oracul:          oracul
        :return:                prepared oracul
        """
        return oracul

    @abstractmethod
    def detect_point(self, point: Point) -> None:
        """
        Detect point for collecting metrics
        :param point:           point
        """
        pass

    @abstractmethod
    def get_result(self, **params) -> float:
        """
        Get result metric
        :param params:          params for evaluating metric
        :return:                result metric
        """
        pass


class MetricMethod(OptimizationMethod):
    """Wrapper over the method, collect provided metrics"""

    def __init__(self, method: OptimizationMethod, metrics: list[Metric]) -> None:
        """
        Constructor of wrapper
        :param method:          wrapped method
        :param metrics:         collecting metrics
        """
        self.method = method
        self.metrics = metrics

    def get_result(self, **params) -> list[float]:
        """
        Get result of metrics
        :param params:          params for evaluating metrics
        :return:                result metric
        """
        return [metric.get_result(**params) for metric in self.metrics]

    def initial_step(self, oracul: Oracul, **params) -> tuple[Point, State]:
        prepared_oracul = oracul
        for metric in self.metrics:
            prepared_oracul = metric.process_oracul(prepared_oracul)

        point, next_state = self.method.initial_step(prepared_oracul, **params)
        for metric in self.metrics:
            metric.detect_point(point)

        return point, next_state

    def step(self, oracul: Oracul, state: State) -> tuple[Point, State]:
        prepared_oracul = oracul
        for metric in self.metrics:
            prepared_oracul = metric.process_oracul(prepared_oracul)

        point, next_state = self.method.step(prepared_oracul, state)
        for metric in self.metrics:
            metric.detect_point(point)

        return point, next_state
