import random
from abc import ABC, abstractmethod
from functools import partial

import numpy as np

from .method_processor import MethodProcessor
from .stop_condition import PrecisionCondition, CountCondition
from ..common import Oracul, State, OptimizationMethod, Point, PointFigure, LineFigure
from ..common.oracul import GradientOracul, LambdaOracul


class RandomMethod(OptimizationMethod):
    """Test method"""

    def initial_step(self, oracul: Oracul, **params) -> tuple[Point, State]:
        dimension = oracul.get_dimension()

        coordinates = [0 for _ in range(dimension - 1)]
        coordinates = coordinates + [oracul.evaluate(Point(np.array(coordinates)))]

        point = Point(np.array(coordinates))
        visual_point = PointFigure(coordinates)

        oracul.evaluate(point)

        state = State(visual_state=[visual_point], parameters=[point])
        return point, state

    def step(self, oracul: Oracul, state: State) -> tuple[Point, State]:
        last_coordinates = state.parameters[0].coordinates
        new_coordinates = [i + random.randint(-20, 20) for i in last_coordinates[:-1]]

        new_coordinates = new_coordinates + [oracul.evaluate(Point(np.array(new_coordinates)))]

        point = Point(np.array(new_coordinates))
        visual_point = PointFigure(new_coordinates)
        visual_line = LineFigure(last_coordinates, new_coordinates)

        oracul.evaluate(point)  # просто так

        state = State(visual_state=[visual_point, visual_line], parameters=[point])
        return point, state


class GoldenRatioMethod(OptimizationMethod):
    _PROPORTION = (1 + np.sqrt(5)) / 2

    def __init__(self):
        self.right_intermediate_dec = None
        self.right_intermediate = None
        self.left_intermediate_dec = None
        self.left_intermediate = None
        self.left_border = None
        self.right_border = None

    def calc_left_intermediate(self) -> float:
        return self.left_border + (self.right_border - self.left_border) / (GoldenRatioMethod._PROPORTION + 1)

    def calc_right_intermediate(self) -> float:
        return self.right_border - (self.right_border - self.left_border) / (GoldenRatioMethod._PROPORTION + 1)

    def calc_temp_res(self) -> Point:
        return Point(np.array([(self.right_border + self.left_border) / 2, 0]))

    def get_state(self):
        return State([LineFigure([self.left_border, 0], [self.right_border, 0]), PointFigure([self.left_border, 0]),
                      PointFigure([self.right_border, 0])], [None],
                     self.left_border - self.right_border)

    def initial_step(self, oracul: Oracul, **params) -> tuple[Point, State]:
        self.left_border = params["a"]
        self.right_border = params["b"]
        self.left_intermediate = self.calc_left_intermediate()
        self.right_intermediate = self.calc_right_intermediate()
        self.left_intermediate_dec = oracul.evaluate(Point(np.array([self.left_intermediate])))
        self.right_intermediate_dec = oracul.evaluate(Point(np.array([self.right_intermediate])))
        return self.calc_temp_res(), self.get_state()

    def step(self, oracul: Oracul, state: State) -> tuple[Point, State]:
        if self.left_intermediate_dec < self.right_intermediate_dec:
            self.right_border = self.right_intermediate
            self.right_intermediate = self.left_intermediate
            self.left_intermediate = self.calc_left_intermediate()
            self.right_intermediate_dec = self.left_intermediate_dec
            self.left_intermediate_dec = oracul.evaluate(Point(np.array([self.left_intermediate])))
        else:
            self.left_border = self.left_intermediate
            self.left_intermediate = self.right_intermediate
            self.right_intermediate = self.calc_right_intermediate()
            self.left_intermediate_dec = self.right_intermediate_dec
            self.right_intermediate_dec = oracul.evaluate(Point(np.array([self.right_intermediate])))
        return self.calc_temp_res(), self.get_state()


class BaseGradientDescent(OptimizationMethod):
    def __init__(self):
        self.x = None
        self.y = None
        self.learning_rate = None

    def get_state(self) -> State:
        return State([PointFigure(np.array(self.x).tolist() + [self.y])], [],
                     float("inf"))

    def get_learning_rate(self, ray, oracul) -> float:
        return self.learning_rate

    def get_temp_res(self):
        return Point(np.append(np.array(self.x), np.array([self.y])))

    def initial_step(self, oracul: GradientOracul, **params) -> tuple[Point, State]:
        self.learning_rate = params["learning_rate"]
        self.x: list[float] = params["x"]
        self.y: float = oracul.evaluate(Point(np.array(self.x, dtype=np.float64)))  # вообще говоря, может и не вычислять
        return self.get_temp_res(), self.get_state()

    def step(self, oracul: GradientOracul, state: State) -> tuple[Point, State]:
        gradient_at_x = oracul.evaluate_gradient(Point(np.array(self.x, np.float64)))
        gradient_at_x = gradient_at_x / np.linalg.norm(gradient_at_x, ord=1)
        self.x = np.array(self.x) - gradient_at_x * self.get_learning_rate(
            gradient_at_x, oracul)
        self.y = oracul.evaluate(Point(np.array(self.x, dtype=np.float64)))
        return self.get_temp_res(), self.get_state()


class GradientDescent(BaseGradientDescent):
    def __init__(self, step_precision=0.05, method=GoldenRatioMethod()):
        super().__init__()
        self.method = method
        self.step_precision = step_precision

    def get_learning_rate(self, ray, oracul):
        point, metrics, anim = MethodProcessor.process(self.method,
                                                       LambdaOracul(lambda rate: oracul.evaluate(
                                                           Point(np.array(self.x) - rate * ray))),
                                                       CountCondition(20),  # Should be PrecisionCondition
                                                       metrics=None, method_params={"a": 0,
                                                                                    "b": self.learning_rate},
                                                       visualize=False)
        return point.coordinates[0]
