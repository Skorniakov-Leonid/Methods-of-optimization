from abc import ABC, abstractmethod
from typing import Tuple, List, Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import Animation
from scipy.optimize import minimize

from ..common import Point, LambdaOracul, PointFigure
from ..common.method import OptimizationMethod, Oracul
from ..common.oracul import MultiLambdaOracul
from ..lab1.method_processor import MethodProcessor
from ..lab1.stop_condition import PrecisionCondition
from ..metric import CallCount, GradientCount, StepCountBeforePrecision
from ..visualization import Animator


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


class ScipyTester(Tester):
    def __init__(self, methods: [str], oraculs: list[tuple[LambdaOracul | MultiLambdaOracul, LambdaOracul | MultiLambdaOracul]],
                 start_point: Point = Point(np.array([100, 200])), eps: float = 1e-3, visualize: bool = True):
        self.methods = methods
        self.oraculs = oraculs
        self.start_point = start_point
        self.eps = eps
        self.visualize = visualize

    def test(self) -> tuple[list[list[list[list[Point]]]], list[list[list[list[Any]]]], list[None]]:
        points = []
        metrics = []
        for method in self.methods:
            part_points = []
            part_metrics = []
            for oracul in self.oraculs:
                plt.figure()
                res = minimize(method=method, fun=oracul[0].func, x0=self.start_point.coordinates,
                               options={"return_all": True, "disp": True, "xatol": True, "fatol": True}, tol=self.eps)
                point = res.x
                if self.visualize:
                    figures = []
                    for point in res.allvecs:
                        figures.append(PointFigure([*point, oracul[1].evaluate(Point(point))]))
                    Animator.animate([figures], oracul[1], oracul[1].get_dimension(), step=0.01, start=[-100, -100],
                                     end=[200, 200])
                    plt.show()
                part_points.append([Point(point)])
                part_metrics.append([res.nfev])
            points.append([part_points])
            metrics.append([part_metrics])
        return points, metrics, [None]
