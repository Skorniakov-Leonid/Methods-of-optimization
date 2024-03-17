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
