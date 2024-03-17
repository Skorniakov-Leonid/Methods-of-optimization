import random

import numpy as np

from ..common import Oracul, State, OptimizationMethod, Point, PointFigure, LineFigure


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
        return self.left_border + 1 / GoldenRatioMethod._PROPORTION * (self.right_border - self.left_border)

    def calc_right_intermediate(self) -> float:
        return self.right_border - 1 / GoldenRatioMethod._PROPORTION * (self.right_border - self.left_border)

    def calc_temp_res(self) -> Point:
        return Point(np.array([(self.right_border + self.left_border) / 2, 0]))

    def get_state(self):
        return State([LineFigure([self.left_border, 0], [self.right_border, 0]), PointFigure([self.left_border, 0]),
                      PointFigure([self.right_border, 0])], [None],
                     self.left_border - self.right_border)

    def initial_step(self, oracul: Oracul, **params) -> tuple[Point, State]:
        self.left_border = params["a"]
        self.right_border = params["b"]
        print(self.left_border, self.right_border)
        self.left_intermediate = self.calc_left_intermediate()
        self.right_intermediate = self.calc_right_intermediate()
        self.left_intermediate_dec = oracul.evaluate(Point(np.array([self.right_intermediate])))
        self.right_intermediate_dec = oracul.evaluate(Point(np.array([self.right_intermediate])))
        return self.calc_temp_res(), self.get_state()

    def step(self, oracul: Oracul, state: State) -> tuple[Point, State]:
        print(self.left_border, self.right_border)
        if self.right_intermediate_dec < self.left_intermediate_dec:
            self.right_border = self.right_intermediate
            self.right_intermediate = self.left_intermediate
            self.left_intermediate = self.calc_left_intermediate()
            self.right_intermediate_dec = self.left_intermediate_dec
            self.left_intermediate_dec = oracul.evaluate(Point(np.array([self.right_intermediate])))
        else:
            self.left_border = self.left_intermediate
            self.left_intermediate = self.right_intermediate
            self.right_intermediate = self.calc_right_intermediate()
            self.left_intermediate_dec = self.right_intermediate_dec
            self.right_intermediate_dec = oracul.evaluate(Point(np.array([self.right_intermediate])))
        return self.calc_temp_res(), self.get_state()

# class GradientDescent(OptimizationMethod):
