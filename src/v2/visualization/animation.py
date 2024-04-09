import itertools
from typing import Optional, Any, Union

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import Animation, ArtistAnimation
from matplotlib.artist import Artist
from matplotlib.axes import Axes

from src.v2.model.meta import Meta
from src.v2.model.method import State
from src.v2.model.oracul import Oracul
from src.v2.visualization.visualization import VisualizationModule, VisualizationMeta

FULL_ANIMATION = {"animate_main": True,
                  "animate_contour": True}


class Animator(VisualizationModule):

    def __init__(self):
        self.oracul: Optional[Oracul] = None
        self.points: list[np.ndarray] = []

    def prepare_oracul(self, oracul: Oracul, **params) -> Oracul:
        self.oracul = oracul
        return oracul

    def process_step(self, state: State, meta: Meta, **params) -> bool:
        self.points.append(state.point.copy())
        return True

    def get_result(self, **params) -> list[tuple[str, Any]]:
        meta = self.meta()
        return [("no_show_" + meta.full_name(), Animator.visualize(self.oracul, self.points, **params))]

    def meta(self) -> VisualizationMeta:
        return VisualizationMeta(name="Animator",
                                 version=f"({self.oracul.get_dimension()}d)",
                                 description="Animation of optimization method's work")

    @staticmethod
    def visualize(oracul: Oracul, points: list[np.ndarray], **params) -> list[Animation]:
        dimension = oracul.get_dimension()
        print(f"[INFO][Animator] Started {dimension}d visualization")
        animations: list[Animation] = []

        default_low_bracket = params.get("animation_low_bracket")
        if default_low_bracket:
            points = points + [default_low_bracket]

        default_high_bracket = params.get("animation_high_bracket")
        if default_high_bracket:
            points = points + [default_high_bracket]

        low_bracket = np.array([min(i) - 0.1 * abs(min(i)) for i in zip(*points)])
        high_bracket = np.array([max(i) + 0.1 * abs(max(i)) for i in zip(*points)])

        point_count = params.get("animation_point_count", 50)
        start_point_count = params.get("animation_start_point_count", 5)
        end_point_count = params.get("animation_end_point_count", 1)

        skip = max(len(points) // (point_count - start_point_count - end_point_count), 1)

        if params.get("animate_contour") and dimension == 3:
            interval = params.get("animation_contour_interval", 250)

            frames: list[list[Artist]] = []

            figure_contour = plt.figure()
            axes = figure_contour.add_subplot()

            contour = Animator.contour(axes, oracul, low_bracket, high_bracket, **params)
            last_point: Optional[np.ndarray] = None
            for index in itertools.chain(range(start_point_count),
                                         range(start_point_count, len(points) - end_point_count, skip),
                                         range(len(points) - end_point_count, len(points))):
                if index >= len(points):
                    break
                point = points[index]
                if index == 0:
                    current_frame = contour
                    current_frame += axes.plot(*point, 'o', color="red")
                else:
                    current_frame = frames[-1][:-1]
                    current_frame += axes.plot(*last_point, 'o', color="blue")
                    current_frame += axes.plot(*point, 'o', color="red")
                frames.append(current_frame)
                last_point = point

            animations.append(ArtistAnimation(
                figure_contour,
                frames,
                interval=interval,
                blit=True,
                repeat=True
            ))

        if params.get("animate_main"):
            interval = params.get("animation_main_interval", 250)
            frames: list[list[Artist]] = []

            figure_main = plt.figure()
            axes = figure_main.add_subplot() if dimension == 2 else figure_main.add_subplot(projection='3d')
            axes.set_xlabel("x")
            axes.set_ylabel("y")
            if dimension == 3:
                axes.set_zlabel("z")

            surface_main = Animator.surface(axes, oracul, low_bracket, high_bracket, **params)

            last_point: Optional[np.ndarray] = None
            for index in itertools.chain(range(start_point_count),
                                         range(start_point_count, len(points) - end_point_count, skip),
                                         range(len(points) - end_point_count, len(points))):
                if index >= len(points):
                    break
                point = points[index]
                evaluated_point = np.concatenate((point, [oracul.evaluate(point)]))
                if index == 0:
                    current_frame = surface_main
                    current_frame += axes.plot(*evaluated_point, 'o', color="red")
                else:
                    current_frame = frames[-1][:-1]
                    current_frame += axes.plot(*last_point, 'o', color="blue")
                    current_frame += axes.plot(*evaluated_point, 'o', color="red")
                frames.append(current_frame)
                last_point = evaluated_point

            animations.append(ArtistAnimation(
                figure_main,
                frames,
                interval=interval,
                blit=True,
                repeat=True
            ))

        return animations

    @staticmethod
    def surface(axes: Axes, oracul: Oracul, low_bracket: np.ndarray, high_bracket: np.ndarray,
                **params) -> list[Artist]:
        step = params.get("animator_main_step", 0.01)

        if oracul.get_dimension() == 2:
            x_grid = np.arange(low_bracket[0], high_bracket[0], (high_bracket[0] - low_bracket[0]) * step)
            y_grid = np.array([oracul.evaluate(np.array([x_val])) for x_val in x_grid])

            return axes.plot(x_grid, y_grid, alpha=0.8)
        else:
            x_stamps = np.arange(low_bracket[0], high_bracket[0], (high_bracket[0] - low_bracket[0]) * step)
            y_stamps = np.arange(low_bracket[1], high_bracket[1], (high_bracket[1] - low_bracket[1]) * step)

            x_grid, y_grid = np.meshgrid(x_stamps, y_stamps)
            z_grid = np.array([[oracul.evaluate(np.array([x_val, y_val])) for x_val in x_stamps] for y_val in y_stamps])

            return [axes.plot_surface(x_grid, y_grid, z_grid, rstride=4, cstride=4, shade=False, alpha=0.5),
                    axes.plot_wireframe(x_grid, y_grid, z_grid, rstride=4, cstride=4, alpha=0.65)]

    @staticmethod
    def contour(axes: Axes, oracul: Oracul, low_bracket: np.ndarray, high_bracket: np.ndarray,
                **params) -> list[Artist]:
        step = params.get("animator_main_step", 0.01)

        x_stamps = np.arange(low_bracket[0], high_bracket[0], (high_bracket[0] - low_bracket[0]) * step)
        y_stamps = np.arange(low_bracket[1], high_bracket[1], (high_bracket[1] - low_bracket[1]) * step)

        x_grid, y_grid = np.meshgrid(x_stamps, y_stamps)
        z_grid = np.array([[oracul.evaluate(np.array([x_val, y_val])) for x_val in x_stamps] for y_val in y_stamps])

        return [axes.contour(x_grid, y_grid, z_grid)]
