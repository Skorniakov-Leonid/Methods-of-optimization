import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..common import OptimizationMethod, Oracul, Point, State


@dataclass
class MetricResult:
    metric_name: str
    method_name: str
    result: float


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
    def detect_step(self, point: Point, state: State) -> None:
        """
        Detect point for collecting metrics
        :param point:           point
        :param state:           state after step
        """
        pass

    @abstractmethod
    def get_result(self, method_name: str = "", **params) -> MetricResult:
        """
        Get result metric
        :param method_name:     name of method
        :param params:          params for evaluating metric
        :return:                result metric
        """
        pass

    @abstractmethod
    def name(self) -> str:
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
        self.metrics = [copy.copy(metric) for metric in metrics]

    def get_result(self, **params) -> list[MetricResult]:
        """
        Get result of metrics
        :param params:          params for evaluating metrics
        :return:                result metric
        """
        return [metric.get_result(self.method.name(), **params) for metric in self.metrics]

    def initial_step(self, oracul: Oracul, visualize: bool = False, **params) -> tuple[Point, State]:
        prepared_oracul = oracul
        for metric in self.metrics:
            prepared_oracul = metric.process_oracul(prepared_oracul)

        point, next_state = self.method.initial_step(prepared_oracul, visualize, **params)
        for metric in self.metrics:
            metric.detect_step(point, next_state)

        return point, next_state

    def step(self, oracul: Oracul, state: State, visualize: bool = False) -> tuple[Point, State]:
        prepared_oracul = oracul
        for metric in self.metrics:
            prepared_oracul = metric.process_oracul(prepared_oracul)

        point, next_state = self.method.step(prepared_oracul, state, visualize)
        for metric in self.metrics:
            metric.detect_step(point, next_state)

        return point, next_state

    def name(self) -> str:
        return "MetricMethod"
