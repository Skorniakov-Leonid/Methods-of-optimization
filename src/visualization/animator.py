import itertools
import math

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation, Animation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
import typing as tp

from ..common import Oracul, Point, Figure


class Animator:
    @staticmethod
    def animate(figures: list[list[Figure]], oracul: tp.Optional[Oracul] = None,
                dimension: tp.Optional[int] = 2, interval: int = 150,
                main_color: str = 'red', common_color: str = 'green',
                start=None, end=None, step: float = 0.01, **params) -> Animation:
        if dimension is None and oracul is None:
            raise ValueError("Unknown dimension for Animator")
        elif oracul is not None and oracul.get_dimension() != dimension:
            raise ValueError("dimension m")
        dimension = dimension if dimension is not None else oracul.get_dimension()
        if dimension not in [2, 3]:
            raise ValueError("Animator support only 2 and 3 dimensions. Current:", dimension)
        figure = plt.figure()
        figc = plt.figure()
        axes = [figure.add_subplot() if dimension == 2 else figure.add_subplot(projection='3d'),
                figc.add_subplot()]
        oracul_surface: list[Artist] = Animator.oracul_to_surface(axes, oracul, start, end, step,
                                                                  **params) if oracul is not None else []

        frames: list[list[Artist]] = []

        show_skip = math.ceil((len(figures) + 1) / 50)

        for index in itertools.chain(range(10), range(10, len(figures), show_skip)):
            if index >= len(figures):
                break
            main_figures = figures[index]
            if index == 0:
                current_frame = oracul_surface.copy()
                current_frame += Animator.figures_to_artists(main_figures, axes, color=main_color)
                frames.append(current_frame)
            else:
                current_frame = frames[-1][:-1]
                current_frame += Animator.figures_to_artists(figures[index - 1], axes, color=common_color)
                current_frame += Animator.figures_to_artists(main_figures, axes, color=main_color)
                frames.append(current_frame)

        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        if dimension == 3:
            axes[0].set_zlabel("z")
        return ArtistAnimation(
            figure,
            frames,
            interval=interval,
            blit=True,
            repeat=True
        )

    @staticmethod
    def figures_to_artists(figures: list[Figure], axes: list[Axes], color: tp.Optional[str] = None) -> list[Artist]:
        main_ax = axes[0]
        cont_ax = axes[1]
        result = []
        result += list(itertools.chain(*[figure.visualize(main_ax, color=color) for figure in figures]))
        result += list(itertools.chain(*[figure.visualize_contur(cont_ax, color=color) for figure in figures]))
        return result

    @staticmethod
    def oracul_to_surface(axes: list[Axes], oracul: Oracul, start=None, end=None, step: float = 0.005, **params) -> list[
        Artist]:
        main_ax = axes[0]
        contur_ax = axes[1]

        result = []
        if start is None:
            start = [-5.0, -5.0]
        if end is None:
            end = [5.0, 5.0]
        if any(map(lambda st, en: en - st < 0, start, end)):
            raise ValueError("Invalid intervals for surface")
        if oracul.get_dimension() == 2:
            x = np.arange(start[0], end[0], (end[0] - start[0]) * step)
            y = np.array([oracul.evaluate(Point(np.array([xval]))) for xval in x])

            result += [main_ax.plot(x, y, alpha=0.8, **params)]
        elif oracul.get_dimension() == 3:
            x = np.arange(start[0], end[0], (end[0] - start[0]) * step)
            y = np.arange(start[1], end[1], (end[1] - start[1]) * step)
            xgrid, ygrid = np.meshgrid(x, y)
            zgrid = np.array([[oracul.evaluate(Point(np.array([xval, yval]))) for xval in x] for yval in y])

            result += [
                main_ax.plot_surface(xgrid, ygrid, zgrid, rstride=4, cstride=4, shade=False, alpha=0.5, **params),
                main_ax.plot_wireframe(xgrid, ygrid, zgrid, rstride=4, cstride=4, alpha=0.65),
                contur_ax.contour(xgrid, ygrid, zgrid, rstride=4, cstride=4)
            ]
        else:
            raise ValueError("Supported oraculs only with 2 or 3 dimensions")

        return result
