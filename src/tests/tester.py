from abc import ABC, abstractmethod

import numpy as np
from matplotlib.animation import Animation

from ..common import Point
from ..common.method import OptimizationMethod, Oracul
from ..lab1.method_processor import MethodProcessor
from ..lab1.stop_condition import PrecisionCondition
from ..metric import CallCount, GradientCount, StepCountBeforePrecision


class Tester(ABC):
    @abstractmethod
    def test(self) -> tuple[list[list[Point | None]], list[list[list[float] | None]], list[Animation | None]]:
        pass


class SimpleTester(Tester):
    def __init__(self, methods: list[OptimizationMethod], oraculs: list[Oracul],
                 start_point: Point = Point(np.array([100, 200])), eps: float = 0.001, learning_rate: float = 300,
                 visualize: bool = True, low_bracket: list[float] = None, high_bracket: list[float] = None):
        self.methods = methods
        self.oraculs = oraculs
        self.eps = eps
        self.start_point = start_point
        self.learning_rate = learning_rate
        self.visualize = visualize
        self.low_bracket = low_bracket
        self.high_bracket = high_bracket

    def test(self) -> tuple[
        list[list[list[list[Point | None]]]], list[list[list[list[list[float] | None]]]], list[list[Animation | None]]]:
        points = []
        metrics = []
        animations = []
        for method in self.methods:
            part_points = []
            part_metrics = []
            for oracul in self.oraculs:
                point, metric, animation = MethodProcessor.process(method, oracul, PrecisionCondition(self.eps),
                                                                   metrics=[CallCount(), GradientCount(),
                                                                            StepCountBeforePrecision(self.eps)],
                                                                   method_params={
                                                                       "x": self.start_point,
                                                                       "learning_rate": self.learning_rate,
                                                                       "eps": self.eps
                                                                   }, visualize=self.visualize,
                                                                   low_bracket=self.low_bracket,
                                                                   high_bracket=self.high_bracket)
                part_points.append([point])
                part_metrics.append([metric])
                animations.append([animation])
            points.append([part_points])
            metrics.append([part_metrics])
        return points, metrics, animations
