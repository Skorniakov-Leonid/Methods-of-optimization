import numpy as np
import typing as tp

from matplotlib.artist import Artist
from matplotlib.axes import Axes

from abc import ABC, abstractmethod

from mpl_toolkits.mplot3d import Axes3D


class Figure(ABC):
    """Class for visualizing figure"""

    @abstractmethod
    def visualize(self, axes: Axes, **params) -> list[Artist]:
        """
        Visualize figure on axes
        :param axes:        axes where figure is visualizing
        :param params:      optional parameters for visualizing
        :return:            visualized lines
        """
        pass


class PointFigure(Figure):
    """Figure - point"""

    def __init__(self, coordinates: list[float], **params) -> None:
        """
        Constructor for point
        :param coordinates: list of coordinates of point
        :param params       optional parameters
        """
        self.coordinates = coordinates
        self.params = params

    def visualize(self, axes: Axes, **params) -> list[Artist]:
        merged_parameters = self.params.copy()
        merged_parameters.update(params)

        return axes.plot(*self.coordinates, 'o', **merged_parameters)


class LineFigure(Figure):
    """Figure - line"""

    def __init__(self, start: list[float], end: list[float], **params) -> None:
        """
        Constructor for line
        :param start:       point of start line
        :param end:         point of end line
        :param params:      optional parameters
        """
        self.start = start
        self.end = end
        self.params = params

    def visualize(self, axes: Axes, **params) -> list[Artist]:
        merged_parameters = self.params.copy()
        merged_parameters.update(params)

        transposed_points = [list(i) for i in zip(*[self.start, self.end])]
        return axes.plot(*transposed_points, **merged_parameters)


class TriangleFigure(Figure):
    """Figure - triangle"""

    def __init__(self, points: list[list[float]], **params) -> None:
        """
        Constructor for triangle
        :param points:      list of points
        :param params:      optional parameters
        """
        points.append(points[0])
        self.points = np.array(points)
        self.params = params

    def visualize(self, axes: Axes, **params) -> list[Artist]:
        merged_parameters = self.params.copy()
        merged_parameters.update(params)

        transposed_points = [list(i) for i in zip(*self.points)]
        if isinstance(axes, Axes3D):
            artist = axes.plot_trisurf(*transposed_points, **merged_parameters)
            return [artist]
        else:
            lines = axes.plot(*transposed_points, **merged_parameters)
            return lines + axes.fill(*transposed_points)


class VectorFigure(Figure):
    """Figure - vector"""

    def __init__(self, start: list[float], direction: tp.Optional[list[float]] = None,
                 end: tp.Optional[list[float]] = None, **params) -> None:
        """
        Constructor for vector
        :param start:       point of start vector
        :param direction:   direction of vector
        :param end:         point of end vector
        :param params:      optional parameters
        """
        if direction is not None and end is not None:
            raise ValueError("Cannot combine direction and error")
        elif direction is not None:
            self.direction = direction
        elif end is not None:
            self.direction = list(map(lambda a, b: a - b, end, start))
        else:
            raise ValueError("direction or end must be not None")
        self.start = start
        self.params = params

    def visualize(self, axes: Axes, **params) -> list[Artist]:
        merged_parameters = self.params.copy()
        merged_parameters.update(params)

        return [axes.quiver(*self.start, *self.direction, **merged_parameters)]
