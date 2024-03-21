from __future__ import annotations

import copy
from abc import ABC, abstractmethod
import typing as tp

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import Animation
from scipy.optimize import minimize

from ..common import Point, LambdaOracul, PointFigure
from ..common.method import OptimizationMethod, Oracul
from ..common.oracul import MultiLambdaOracul, NoiseOracul
from ..lab1.method_processor import MethodProcessor
from ..lab1.stop_condition import PrecisionCondition, StopCondition
from ..metric import CallCount, GradientCount, PrecisionCount, Metric
from ..metric.metric import MetricResult
from ..visualization import Animator


class Tester:
    @staticmethod
    def test(methods: list[OptimizationMethod], oraculs: list[Oracul], metrics: list[Metric],
             stop_condition: StopCondition, start_point: tp.Optional[list[float]] = None,
             learning_rate: float = 300, visualize: bool = True, noise: float = 0.0):
        all_points = []
        for index, oracul in enumerate(oraculs):
            header = [metric.name() for metric in metrics]
            row_header = [method.name() for method in methods]
            columns: list[list[tp.Any]] = []

            metric_results: list[list[MetricResult]] = []
            points: list[list[float]] = []

            if noise > 10 ** -10:
                oracul = NoiseOracul(oracul, -noise, noise)
            for method in methods:
                method = copy.copy(method)
                point, results, _ = MethodProcessor.process(method, oracul, stop_condition, metrics,
                                                            visualize=False,
                                                            method_params={"start_point": start_point,
                                                                           "learning_rate": learning_rate})
                metric_results += [results]
                points += [point]
                columns += [[res.result for res in results]]

            N = len(methods)
            fig, ax = plt.subplots(figsize=(17, 3 + N / 2.5))

            table = plt.table(cellText=columns,
                              rowLabels=row_header,
                              colLabels=header,
                              cellLoc='center',
                              loc='center right')
            table.scale(0.95, 2.5)
            table.auto_set_font_size(False)
            table.set_fontsize(16)
            ax.axis('off')
            ax.set_title(index + 1, weight='bold', size=14, color='k')
            all_points += [points]
            plt.savefig("table.png", dpi=200, bbox_inches='tight')

        return all_points


class OldTester(ABC):
    @abstractmethod
    def test(self) -> tuple[list[list[Point | None]], list[list[list[float] | None]], list[Animation | None]]:
        pass


class SimpleOldTester(OldTester):
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
                                                                            PrecisionCount(self.eps)],
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


class ScipyOldTester(OldTester):
    def __init__(self, methods: [str],
                 oraculs: list[tuple[LambdaOracul | MultiLambdaOracul, LambdaOracul | MultiLambdaOracul]],
                 start_point: Point = Point(np.array([100, 200])), eps: float = 1e-3, visualize: bool = True):
        self.methods = methods
        self.oraculs = oraculs
        self.start_point = start_point
        self.eps = eps
        self.visualize = visualize

    def test(self) -> tuple[list[list[list[list[Point]]]], list[list[list[list[tp.Any]]]], list[None]]:
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
