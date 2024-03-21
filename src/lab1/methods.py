import numpy as np
import typing as tp

from scipy.optimize import minimize

from .method_processor import MethodProcessor
from .stop_condition import PrecisionCondition
from ..common import Oracul, State, OptimizationMethod, Point, PointFigure, LineFigure
from ..common.oracul import GradientOracul, LambdaOracul


class GoldenRatioMethod(OptimizationMethod):
    """Optimization method using golden ratio"""
    _PROPORTION = (1 + np.sqrt(5)) / 2

    right_intermediate_dec: tp.Optional[float] = None
    right_intermediate: tp.Optional[float] = None
    left_intermediate_dec: tp.Optional[float] = None
    left_intermediate: tp.Optional[float] = None
    left_border: tp.Optional[float] = None
    right_border: tp.Optional[float] = None

    def calc_left_intermediate(self) -> float:
        return self.left_border + (self.right_border - self.left_border) / (GoldenRatioMethod._PROPORTION + 1)

    def calc_right_intermediate(self) -> float:
        return self.right_border - (self.right_border - self.left_border) / (GoldenRatioMethod._PROPORTION + 1)

    def calc_temp_res(self) -> Point:
        return Point(np.array([(self.right_border + self.left_border) / 2, 0]))

    def get_state(self, visualize: bool = False):
        if not visualize:
            return State(
                [],
                [],
                self.right_border - self.left_border
            )
        return State(
            [
                LineFigure(
                    [self.left_border, 0],
                    [self.right_border, 0]
                ),
                PointFigure([self.left_border, 0]),
                PointFigure([self.right_border, 0])
            ],
            [],
            self.right_border - self.left_border
        )

    def initial_step(self, oracul: Oracul, visualize: bool = False, **params) -> tuple[Point, State]:
        self.left_border = params["left"]
        self.right_border = params["right"]
        self.left_intermediate = self.calc_left_intermediate()
        self.right_intermediate = self.calc_right_intermediate()
        self.left_intermediate_dec = oracul.evaluate(Point(np.array([self.left_intermediate])))
        self.right_intermediate_dec = oracul.evaluate(Point(np.array([self.right_intermediate])))
        return self.calc_temp_res(), self.get_state(visualize)

    def step(self, oracul: Oracul, state: State, visualize: bool = False) -> tuple[Point, State]:
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
        return self.calc_temp_res(), self.get_state(visualize)

    def name(self) -> str:
        return "GoldenRation"


class BaseGradientDescent(OptimizationMethod):
    """Class for gradient descent method with fixed rate"""
    learning_rate: tp.Optional[float] = None
    prev_x: tp.Optional[list[float]] = None
    x: tp.Optional[list[float]] = None
    y: tp.Optional[float] = None

    def get_state(self, visualize: bool = False) -> State:
        if not visualize:
            return State(
                [],
                [],
                self.get_precision()
            )
        return State(
            [PointFigure(np.array(self.x).tolist() + [self.y])],
            [],
            self.get_precision()
        )

    def get_learning_rate(self, ray, oracul) -> float:
        return self.learning_rate

    def get_temp_res(self):
        return Point(np.append(np.array(self.x), np.array([self.y])))

    def initial_step(self, oracul: GradientOracul, visualize: bool = False, **params) -> tuple[Point, State]:
        self.learning_rate = params["learning_rate"]
        self.x = params["start_point"]
        self.prev_x = None
        self.y = oracul.evaluate(
            Point(np.array(self.x, dtype=np.float64))
        )  # вообще говоря, может и не вычислять
        return self.get_temp_res(), self.get_state(visualize)

    def step(self, oracul: GradientOracul, state: State, visualize: bool = False) -> tuple[Point, State]:
        gradient_at_x = oracul.evaluate_gradient(Point(np.array(self.x, np.float64)))
        gradient_at_x = np.float64(gradient_at_x) / np.linalg.norm(gradient_at_x, ord=2)
        self.prev_x = self.x
        self.x = np.array(self.x) - gradient_at_x * self.get_learning_rate(
            gradient_at_x, oracul)
        self.y = oracul.evaluate(Point(np.array(self.x, dtype=np.float64)))
        return self.get_temp_res(), self.get_state(visualize)

    def get_precision(self):
        return float("inf") if self.prev_x is None else np.sqrt(np.sum(np.square(self.x - self.prev_x)))

    def name(self) -> str:
        return "BaseGradient({})".format(self.learning_rate)


class GradientDescent(BaseGradientDescent):
    """Class for gradient descent method"""

    def __init__(self, aprox_dec=0.0001, method=GoldenRatioMethod()):
        super().__init__()
        self.method = method
        self.eps = aprox_dec

    def get_learning_rate(self, ray, oracul):
        point, _, _ = MethodProcessor.process(
            self.method,
            LambdaOracul(
                lambda rate: oracul.evaluate(Point(np.array(self.x) - rate * ray))
            ),
            PrecisionCondition(self.eps),
            metrics=None,
            method_params={"left": 0, "right": self.learning_rate},
            visualize=False)
        return np.float64(point.coordinates[0])

    def name(self) -> str:
        return "Gradient({})".format(self.method.name()[0])


class CoordinateDescent(OptimizationMethod):
    """Class for coordinate descent method"""
    x_dec: tp.Optional[float] = None
    dim_num: tp.Optional[int] = None
    x: tp.Optional[np.ndarray] = None
    temp_dim: int = 0

    def __init__(self, learning_rate: float = 300, eps: float = 0.01) -> None:
        self.learning_rate = learning_rate
        self.step_len = learning_rate
        self.precision = eps

    def get_state(self, visualize: bool = False) -> State:
        if not visualize:
            return State(
                [],
                [],
                self.step_len
            )
        return State(
            [PointFigure(np.array(self.x).tolist() + [self.x_dec])],
            [],
            self.step_len
        )

    def get_temp_res(self):
        return Point(np.append(np.array(self.x), np.array([self.x_dec])))

    def initial_step(self, oracul: Oracul, visualize: bool = False, **params) -> tuple[Point, State]:
        self.x = np.array(params["start_point"], np.float64)
        self.x_dec = oracul.evaluate(Point(self.x))
        self.dim_num = len(self.x)
        self.temp_dim = 0
        return self.get_temp_res(), self.get_state(visualize)

    def step(self, oracul: Oracul, state: State, visualize: bool = False) -> tuple[Point, State]:
        success = False
        checked_dim = 0
        while self.step_len > self.precision and not success:
            temp_step = np.zeros(self.dim_num, np.float64)
            temp_step[self.temp_dim] = self.step_len
            temp_dec = oracul.evaluate(Point(self.x + temp_step))
            if self.x_dec > temp_dec:
                self.x[self.temp_dim] += self.step_len
                self.x_dec = temp_dec
                success = True
            else:
                temp_step[self.temp_dim] = -self.step_len
                temp_dec = oracul.evaluate(Point(self.x + temp_step))
                if self.x_dec > temp_dec:
                    self.x[self.temp_dim] -= self.step_len
                    self.x_dec = temp_dec
                    success = True
            self.temp_dim = (self.temp_dim + 1) % self.dim_num
            checked_dim = (checked_dim + 1) % self.dim_num
            if checked_dim == 0 and not success:
                self.step_len /= 2
        return self.get_temp_res(), self.get_state(visualize)

    def name(self) -> str:
        return "Coordinate({})".format(self.learning_rate)


class DichotomyMethod(OptimizationMethod):
    """Class for dichotomy method"""
    left_border = None
    right_border = None
    eps = None

    def initial_step(self, oracul: Oracul, visualize: bool = False, **params) -> tuple[Point, State]:
        self.left_border = params["left"]
        self.right_border = params["right"]
        self.eps = 1e-6
        return self.calc(), self.get_state(visualize)

    def get_state(self, visualize: bool = False) -> State:
        if not visualize:
            return State(
                [],
                [None],
                self.right_border - self.left_border)
        return State(
            [
                LineFigure(
                    [self.left_border, 0],
                    [self.right_border, 0]
                ),
                PointFigure([self.left_border, 0]),
                PointFigure([self.right_border, 0])
            ],
            [None],
            self.right_border - self.left_border)

    def calc(self) -> Point:
        return Point(np.array([(self.left_border + self.right_border) / 2]))

    def step(self, oracul: Oracul, state: State, visualize: bool = False) -> tuple[Point, State]:
        c = (self.left_border + self.right_border) / 2
        if oracul.evaluate(
                Point(np.array([c - self.eps], dtype=np.float64))
        ) < oracul.evaluate(Point(np.array([c + self.eps], dtype=np.float64))):
            self.right_border = c
        else:
            self.left_border = c
        return self.calc(), self.get_state(visualize)

    def name(self) -> str:
        return "Dichotomy"


class NMMethod(OptimizationMethod):
    """Class for Nelder Mead method"""
    points: tp.Optional[list[np.ndarray]] = None
    iterator: int = 0

    def __init__(self, eps: float) -> None:
        """
        Constructor for Nelder Mead method
        :param eps:     required eps
        """
        self.eps: float = eps

    def initial_step(self, oracul: Oracul, visualize: bool = False, **params) -> tuple[Point, State]:
        def func(coordinates: list[float]) -> float:
            return oracul.evaluate(Point(np.array(coordinates)))

        start_point = params["start_point"] if 'start_point' in params else [0 for _ in range(oracul.get_dimension() - 1)]

        self.points = minimize(
            method="Nelder-Mead",
            fun=func,
            x0=start_point,
            options={"return_all": True, "disp": False, "xatol": True, "fatol": True},
            tol=self.eps
        ).allvecs

        self.iterator = min(self.iterator + 1, len(self.points) - 1)
        point_coordinates = self.points[self.iterator].tolist()
        point_coordinates += [oracul.evaluate(Point(point_coordinates))]
        eps = 0 if self.iterator == (len(self.points) - 1) else float('inf')
        if visualize:
            state = State(
                [PointFigure(point_coordinates)],
                [],
                eps
            )
        else:
            state = State([], [], eps)
        return Point(point_coordinates), state

    def step(self, oracul: Oracul, state: State, visualize: bool = False) -> tuple[Point, State]:
        self.iterator = min(self.iterator + 1, len(self.points) - 1)
        last_point_coordinates = self.points[max(0, self.iterator - 1)]
        point_coordinates = self.points[self.iterator].tolist()
        point_coordinates += [oracul.evaluate(Point(point_coordinates))]
        eps = 0 if self.iterator == (len(self.points) - 1) else float('inf')
        if visualize:
            state = State(
                [
                    LineFigure(
                        last_point_coordinates.tolist() + [oracul.evaluate(Point(last_point_coordinates))],
                        point_coordinates
                    ),
                    PointFigure(point_coordinates)
                ],
                [],
                eps
            )
        else:
            state = State([], [], eps)
        return Point(point_coordinates), state

    def name(self) -> str:
        return "NM({})".format(self.eps)
