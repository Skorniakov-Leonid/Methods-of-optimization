import random
from functools import partial
from inspect import signature
from typing import Callable

import numpy as np
import typing as tp
import sympy
from numpy import ndarray
import numdifftools as nd


from src.v2.model.function_interpretation import FunctionInterpretation
from src.v2.model.loss_function import LossFunction
from src.v2.model.oracul import Oracul, OraculMeta, EpochState


class LinearWrapper(Oracul):
    def __init__(self, oracul: Oracul, x: np.ndarray, ray: np.ndarray, state: EpochState) -> None:
        """
        Constructor for lambda oracul
        :param func:        lambda that generate oracul
        """
        self.oracul = oracul
        self.x = x
        self.ray = ray
        self.state = state

    def evaluate(self, point: np.ndarray, state: EpochState, **params) -> float:
        return self.oracul.evaluate(self.x - point * self.ray, self.state)

    def evaluate_gradient(self, point: np.ndarray, **params) -> ndarray:
        return self.oracul.evaluate_gradient(self.x - point * self.ray)

    def get_dimension(self) -> int:
        return 2

    def meta(self, **params) -> OraculMeta:
        return OraculMeta(name="LambdaOracul",
                          description="Oracul based on lambda")


class LambdaOracul(Oracul):
    """Class for oracul from lambda"""

    def __init__(self, func: Callable[..., float]) -> None:
        """
        Constructor for lambda oracul
        :param func:        lambda that generate oracul
        """
        self.func = func
        self.dimension = len(signature(func).parameters) + 1

    def evaluate(self, point: np.ndarray, state: EpochState, **params) -> float:
        return self.func(*point)

    def get_dimension(self) -> int:
        return len(signature(self.func).parameters) + 1

    def meta(self, **params) -> OraculMeta:
        return OraculMeta(name="LambdaOracul",
                          description="Oracul based on lambda")


class SymbolOracul(Oracul):
    """Oracul that automatically calculates gradients and hessians through sympy"""

    def __init__(self, func, dimensions_order: list[str]):
        self.func = func
        self.dimensions_order = dimensions_order
        self.grad = np.array([])
        for i in self.dimensions_order:
            self.grad = np.append(self.grad, sympy.diff(self.func, i))
        self.hes = []
        for i in self.dimensions_order:
            row_res = []
            for j in self.dimensions_order:
                row_res.append(sympy.diff(sympy.diff(self.func, i), j))
            self.hes.append(np.array(row_res))
        self.hes = np.array(self.hes)

    def map_dim_to_point(self, point: np.ndarray) -> dict:
        return dict(zip(self.dimensions_order, point))

    def evaluate(self, point: np.ndarray, state: EpochState, **params) -> float:
        return np.float64(self.func.subs(self.map_dim_to_point(point)))

    def evaluate_gradient(self, point: np.ndarray, state: EpochState, **params) -> np.ndarray:
        res = [np.float64(i.subs(self.map_dim_to_point(point))) for i in self.grad]
        return np.array(res, dtype=np.float64)

    def evaluate_hessian(self, point: np.ndarray, state: EpochState, **params) -> np.ndarray:
        res = []
        for i in self.hes:
            row_res = []
            for j in i:
                row_res.append(np.float64(j.subs(self.map_dim_to_point(point))))
            res.append(np.array(row_res))
        return np.array(res)

    def get_dimension(self) -> int:
        return len(self.dimensions_order) + 1

    def meta(self, **params) -> OraculMeta:
        return OraculMeta(name="SymbolOracul",
                          description="Oracul based on sympy expression")


class PoweredSumOracul(Oracul):
    """Class for oracul generated by coefficients"""

    def __init__(self, coefficients: list[tuple[float, float]]) -> None:
        """
        Constructor for powered sum oracul
        :param coefficients:    coefficients in style [(coefficient of x, power of x), (coefficient of y, power of y)...]
        """
        self.coefficients = coefficients

    def evaluate(self, point: np.ndarray, state: EpochState, **params) -> float:
        res = np.float64(0)
        for coordinate, coefficient, power in zip(point, self.coefficients):
            res += coefficient * (np.float64(coordinate) ** power)
        for i in range(len(self.coefficients)):
            res += self.coefficients[i][0] * (np.float64(point[i]) ** self.coefficients[i][1])
        return res

    def evaluate_gradient(self, point: np.ndarray, state: EpochState, **params) -> np.ndarray:
        res = np.zeros(len(self.coefficients), dtype=np.float64)
        for i in range(len(self.coefficients)):
            res[i] = (np.float64(point[i]) ** np.float64(self.coefficients[i][1] - 1)) * \
                     self.coefficients[i][0] * \
                     self.coefficients[i][1]
        return res

    def get_dimension(self) -> int:
        return len(self.coefficients) + 1

    def meta(self, **params) -> OraculMeta:
        return OraculMeta(name="PoweredSumOracul",
                          description="Oracul described by powers and coefficients")


class MinimisingOracul(Oracul):
    def __init__(self, minimization_function: LossFunction, interpretation: FunctionInterpretation, data: np.ndarray,
                 crop_size: int, **params) -> None:
        self.minimization_function = minimization_function
        self.interpretation = interpretation
        self.data = data
        self.crop_size = crop_size

    def evaluate(self, point: np.ndarray, state: EpochState, **params) -> float:
        if state is None:
            raise ValueError("State is None!")
        points = np.array([self.data[i] for i in state.points])
        interpreted_points = np.array([self.interpretation.interpretate_data(p) for p in points])
        points = np.array([p[:-1] for p in points])
        return self.minimization_function.eval(interpreted_points, self.interpretation.evaluate(point, points), point)

    def evaluate_gradient(self, point: np.ndarray, state: EpochState, **params) -> np.ndarray:
        gradients = [nd.Gradient(partial(self.evaluate, state=EpochState(points=[p])))(point) for p in state.points]
        return np.array([sum(i) / len(gradients) for i in zip(*gradients)])

    def next_state(self, state: EpochState = None, **params) -> EpochState:
        if state is None:
            return EpochState(0, 0, list(random.sample(range(len(self.data)), self.crop_size)), [])

        all_points = set(range(len(self.data)))
        processed_points = set(state.points + state.previous_points)

        not_processed_points = list(all_points - processed_points)

        epoch = state.epoch
        step = state.step + 1
        if len(not_processed_points) == 0:
            epoch += 1
            step = 0
            processed_points = np.empty(0)
            not_processed_points = list(all_points)

        next_points = [not_processed_points[i] for i in random.sample(
            range(len(not_processed_points)),
            min(self.crop_size, len(not_processed_points)))]

        return EpochState(epoch, step, next_points, list(processed_points))

    def get_dimension(self) -> int:
        return self.interpretation.get_dimension()

    def all_points(self, **params) -> EpochState:
        return EpochState(-1, -1, list(range(len(self.data))), [])

    def meta(self, **params) -> OraculMeta:
        return OraculMeta(name="MinimisingOracul",
                          description="Oracul based on minimization and interpretation functions")

    def get_interpretation(self) -> FunctionInterpretation:
        return self.interpretation

    def get_data(self) -> tp.Optional[np.ndarray]:
        return self.data
