from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.v2.impl.conditions import PrecisionCondition
from src.v2.impl.oraculs import LambdaOracul
from src.v2.model.method import OptimizationMethod, State, MethodMeta
from src.v2.model.oracul import Oracul
from src.v2.runner.runner import Runner


@dataclass
class CoordinateDescentState(State):
    dim_num: int = 0
    temp_dim: int = 0
    dec: float = 0


class CoordinateDescent(OptimizationMethod):
    """Class for coordinate descent method"""

    def __init__(self, learning_rate: float = 300, eps: float = 0.01) -> None:
        self.learning_rate = learning_rate
        self.precision = eps

    def initial_step(self, oracul: Oracul, point: np.ndarray, **params) -> CoordinateDescentState:
        state = CoordinateDescentState(point, self.learning_rate)
        state.dim_num = len(point)
        state.temp_dim = 0
        state.dec = oracul.evaluate(point)
        return state

    def step(self, oracul: Oracul, state: CoordinateDescentState, **params) -> CoordinateDescentState:
        success = False
        checked_dim = 0
        while state.eps > self.precision and not success:
            temp_step = np.zeros(state.dim_num, np.float64)
            temp_step[state.temp_dim] = state.eps
            temp_dec = oracul.evaluate(state.point + temp_step)
            if state.dec > temp_dec:
                state.point[state.temp_dim] += state.eps
                state.dec = temp_dec
                success = True
            else:
                temp_step[state.temp_dim] = -state.eps
                temp_dec = oracul.evaluate(state.point + temp_step)
                if state.dec > temp_dec:
                    state.point[state.temp_dim] -= state.eps
                    state.dec = temp_dec
                    success = True
            state.temp_dim = (state.temp_dim + 1) % state.dim_num
            checked_dim = (checked_dim + 1) % state.dim_num
            if checked_dim == 0 and not success:
                state.eps /= 2
        return state

    def meta(self, **params) -> MethodMeta:
        return MethodMeta(name="CoordinateDescent",
                          version=f"({self.learning_rate})",
                          description="Method of optimization using coordinate descent")


@dataclass
class GoldenRatioState(State):
    right_intermediate_dec: float = 0
    right_intermediate: float = 0
    left_intermediate_dec: float = 0
    left_intermediate: float = 0
    left_border: float = 0
    right_border: float = 0


class GoldenRatioMethod(OptimizationMethod):
    """Optimization method using golden ratio"""
    _PROPORTION = (1 + np.sqrt(5)) / 2

    @staticmethod
    def calc_left_intermediate(state: GoldenRatioState) -> float:
        return state.left_border + (state.right_border - state.left_border) / (GoldenRatioMethod._PROPORTION + 1)

    @staticmethod
    def calc_right_intermediate(state: GoldenRatioState) -> float:
        return state.right_border - (state.right_border - state.left_border) / (GoldenRatioMethod._PROPORTION + 1)

    def initial_step(self, oracul: Oracul, point: np.ndarray, **params) -> GoldenRatioState:
        left_border = params.get("left", -10000000)
        right_border = params.get("right", 10000000)

        state = GoldenRatioState(np.array([(right_border + left_border) / 2]), right_border - left_border)
        state.left_border = left_border
        state.right_border = right_border
        state.left_intermediate = GoldenRatioMethod.calc_left_intermediate(state)
        state.right_intermediate = GoldenRatioMethod.calc_right_intermediate(state)
        state.left_intermediate_dec = oracul.evaluate(np.array([state.left_intermediate]))
        state.right_intermediate_dec = oracul.evaluate(np.array([state.right_intermediate]))
        return state

    def step(self, oracul: Oracul, state: GoldenRatioState, **params) -> GoldenRatioState:
        if state.left_intermediate_dec < state.right_intermediate_dec:
            state.right_border = state.right_intermediate
            state.right_intermediate = state.left_intermediate
            state.left_intermediate = GoldenRatioMethod.calc_left_intermediate(state)
            state.right_intermediate_dec = state.left_intermediate_dec
            state.left_intermediate_dec = oracul.evaluate(np.array([state.left_intermediate]))
        else:
            state.left_border = state.left_intermediate
            state.left_intermediate = state.right_intermediate
            state.right_intermediate = GoldenRatioMethod.calc_right_intermediate(state)
            state.left_intermediate_dec = state.right_intermediate_dec
            state.right_intermediate_dec = oracul.evaluate(np.array([state.right_intermediate]))
        state.point = np.array([(state.right_border + state.left_border) / 2])
        state.eps = state.right_border - state.left_border
        return state

    def meta(self, **params) -> MethodMeta:
        return MethodMeta(name="GoldenRatioMethod",
                          description="Method of optimization using golden ratio")


@dataclass
class GradientDescentState(State):
    prev_point: Optional[np.ndarray] = None


class GradientDescent(OptimizationMethod):
    @staticmethod
    def get_precision(state: GradientDescentState):
        return float("inf") if state.prev_point is None else np.linalg.norm(state.point - state.prev_point)

    def __init__(self, learning_rate: float = 300, method=GoldenRatioMethod(), aprox_dec=0.0001) -> None:
        self.learning_rate = learning_rate
        self.method = method
        self.eps = aprox_dec

    def initial_step(self, oracul: Oracul, point: np.ndarray, **params) -> GradientDescentState:
        state = GradientDescentState(point, float('inf'))
        state.prev_point = None
        return state

    def step(self, oracul: Oracul, state: GradientDescentState, **params) -> GradientDescentState:
        gradient_at_x = oracul.evaluate_gradient(state.point)
        gradient_at_x = gradient_at_x / np.linalg.norm(gradient_at_x, ord=2)
        state.prev_point = state.point
        state.point = state.point - gradient_at_x * self.get_learning_rate(state, gradient_at_x, oracul)
        state.eps = GradientDescent.get_precision(state)
        return state

    def get_learning_rate(self, state: GradientDescentState, ray: np.ndarray, oracul: Oracul) -> float:
        data = Runner.run_pipeline(
            self.method,
            LambdaOracul(
                lambda rate: oracul.evaluate(state.point - rate * ray)
            ),
            np.array([0]),
            [PrecisionCondition(self.eps)],
            left=0,
            right=self.learning_rate)
        return data[0][1]

    def meta(self, **params) -> MethodMeta:
        return MethodMeta(name="GradientDescent",
                          version=f"({self.learning_rate},{self.method.meta().full_name()},eps={self.eps})",
                          description="Method of optimization using gradient descent")