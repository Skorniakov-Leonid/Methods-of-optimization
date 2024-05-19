from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from src.v2.impl.conditions import PrecisionCondition
from src.v2.impl.oraculs import LambdaOracul, LinearWrapper
from src.v2.model.method import OptimizationMethod, State, MethodMeta
from src.v2.model.oracul import Oracul
from src.v2.runner.runner import Runner


@dataclass
class CoordinateDescentState(State):
    precision: float = 1.0e-3
    dim_num: int = 0
    temp_dim: int = 0
    dec: float = 0


class CoordinateDescent(OptimizationMethod):
    """Class for coordinate descent method"""

    def __init__(self, learning_rate: float = 300, aprox_dec: float = 1e-4) -> None:
        self.learning_rate = learning_rate
        self.precision = aprox_dec

    def initial_step(self, oracul: Oracul, point: np.ndarray, **params) -> CoordinateDescentState:
        state = CoordinateDescentState(point, self.learning_rate)
        state.epoch_state = oracul.next_state()
        state.precision = params.get("precision", 1.0e-3)
        state.dim_num = len(point)
        state.temp_dim = 0
        state.dec = oracul.evaluate(point, state.epoch_state)
        return state

    def step(self, oracul: Oracul, state: CoordinateDescentState, **params) -> CoordinateDescentState:
        success = False
        checked_dim = 0
        while state.eps > self.precision and not success:
            temp_step = np.zeros(state.dim_num, np.float64)
            temp_step[state.temp_dim] = state.eps
            temp_dec = oracul.evaluate(state.point + temp_step, state.epoch_state)
            if state.dec > temp_dec:
                state.point[state.temp_dim] += state.eps
                state.dec = temp_dec
                success = True
            else:
                temp_step[state.temp_dim] = -state.eps
                temp_dec = oracul.evaluate(state.point + temp_step, state.epoch_state)
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
        state.epoch_state = oracul.next_state()
        state.left_border = left_border
        state.right_border = right_border
        state.left_intermediate = GoldenRatioMethod.calc_left_intermediate(state)
        state.right_intermediate = GoldenRatioMethod.calc_right_intermediate(state)
        state.left_intermediate_dec = oracul.evaluate(np.array([state.left_intermediate]), state.epoch_state)
        state.right_intermediate_dec = oracul.evaluate(np.array([state.right_intermediate]), state.epoch_state)
        return state

    def step(self, oracul: Oracul, state: GoldenRatioState, **params) -> GoldenRatioState:
        if state.left_intermediate_dec < state.right_intermediate_dec:
            state.right_border = state.right_intermediate
            state.right_intermediate = state.left_intermediate
            state.left_intermediate = GoldenRatioMethod.calc_left_intermediate(state)
            state.right_intermediate_dec = state.left_intermediate_dec
            state.left_intermediate_dec = oracul.evaluate(np.array([state.left_intermediate]), state.epoch_state)
        else:
            state.left_border = state.left_intermediate
            state.left_intermediate = state.right_intermediate
            state.right_intermediate = GoldenRatioMethod.calc_right_intermediate(state)
            state.left_intermediate_dec = state.right_intermediate_dec
            state.right_intermediate_dec = oracul.evaluate(np.array([state.right_intermediate]), state.epoch_state)
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
        state.epoch_state = oracul.next_state()
        state.prev_point = None
        return state

    def step(self, oracul: Oracul, state: GradientDescentState, **params) -> GradientDescentState:
        gradient_at_x = oracul.evaluate_gradient(state.point, state.epoch_state)
        gradient_at_x = gradient_at_x / np.linalg.norm(gradient_at_x, ord=2)
        state.prev_point = state.point
        state.point = state.point - gradient_at_x * self.get_learning_rate(state, gradient_at_x, oracul)
        state.eps = GradientDescent.get_precision(state)
        return state

    def get_learning_rate(self, state: GradientDescentState, ray: np.ndarray, oracul: Oracul) -> float:
        data = Runner.run_pipeline(
            self.method,
            LambdaOracul(
                lambda rate: oracul.evaluate(state.point - rate * ray, state.epoch_state)
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


@dataclass
class ScipyMethodState(State):
    precision: float = 1.0e-3
    points: Optional[list[np.ndarray]] = None
    index: int = 0


class ScipyMethod(OptimizationMethod):
    def __init__(self, method: str, **options) -> None:
        self.method = method.lower()
        self.options = {"options": {"return_all": True}}
        self.options.update(options)

    def initial_step(self, oracul: Oracul, point: np.ndarray, **params) -> ScipyMethodState:
        state = ScipyMethodState(point, float('inf'))
        state.epoch_state = oracul.next_state()
        state.precision = params.get("precision", 1e-3)
        state.points = minimize(
            method=self.method,
            fun=partial(oracul.evaluate, state=state.epoch_state),
            x0=point,
            tol=state.precision,
            jac=partial(oracul.evaluate_gradient, state=state.epoch_state) if self.method not in
                                                                              ('nelder-mead', 'powell', 'cobyla') else None,
            hess=partial(oracul.evaluate_hessian, state=state.epoch_state) if self.method in (
                'newton-cg', 'dogleg', 'trust-ncg', 'trust-constr', 'trust-krylov', 'trust-exact', '_custom') else None,
            **self.options
        ).allvecs
        state.index = -1
        return state

    def step(self, oracul: Oracul, state: ScipyMethodState, **params) -> ScipyMethodState:
        state.index = min(state.index + 1, len(state.points) - 1)
        state.point = state.points[state.index]
        if state.index == len(state.points) - 1:
            state.eps = state.precision
        return state

    def meta(self, **params) -> MethodMeta:
        return MethodMeta(name="ScipyMethod",
                          version=f"({self.method})",
                          description="Method of optimization using scipy.optimize")


@dataclass
class NewtonState(State):
    prev_point: Optional[np.ndarray] = None


class NewtonBase(OptimizationMethod):
    """Newton method. Fixed learning rate"""

    learning_rate = None

    def __init__(self, learning_rate: float = 1) -> None:
        self.learning_rate = learning_rate

    def initial_step(self, oracul: Oracul, point: np.ndarray, **params) -> NewtonState:
        state = NewtonState(point=point, eps=float('inf'))
        state.epoch_state = oracul.next_state()
        state.prev_point = None
        return state

    def step(self, oracul: Oracul, state: NewtonState, **params) -> NewtonState:
        state.prev_point = state.point
        hess = oracul.evaluate_hessian(state.point, state.epoch_state)
        try:
            inverted = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            inverted = np.linalg.inv(hess + np.eye(hess.shape[0]) * 1e-11)
        ray = np.dot(inverted,
                     oracul.evaluate_gradient(state.point, state.epoch_state))
        state.point = (np.array(state.point, dtype=np.float64) - self.get_learning_rate(state.point, ray, oracul, state)
                       * ray)
        state.eps = self.get_precision(state)
        return state

    def meta(self, **params) -> MethodMeta:
        return MethodMeta(name="NewtonBase",
                          version=f"({self.learning_rate})",
                          description="Method of optimization using Newton with fixed rate")

    def get_learning_rate(self, point: np.ndarray, ray: np.ndarray, oracul: Oracul, state: State):
        return self.learning_rate

    @staticmethod
    def get_precision(state: NewtonState):
        return float("inf") if state.prev_point is None else np.sqrt(np.sum(np.square(state.point - state.prev_point)))


class Newton(NewtonBase):

    def __init__(self, learning_rate: float = 3, method=GoldenRatioMethod(), aprox_dec=0.0001) -> None:
        super().__init__(learning_rate)
        self.method = method
        self.eps = aprox_dec

    def get_learning_rate(self, point: np.ndarray, ray: np.ndarray, oracul: Oracul, state: State) -> float:
        data = Runner.run_pipeline(
            self.method,
            LinearWrapper(oracul, point, ray, state.epoch_state),
            np.array([1]),
            [PrecisionCondition(self.eps)],
            left=0,
            right=self.learning_rate)
        return data[0][1]

    def meta(self, **params) -> MethodMeta:
        return MethodMeta(name="Newton",
                          version=f"({self.learning_rate},{self.method.meta().full_name()},eps={self.eps})",
                          description="Method of optimization using Newton with changing rate ")


@dataclass
class WolfeState(State):
    fj_old: Optional[float] = None
    fk: Optional[float] = None
    alpha_old: Optional[np.ndarray] = None
    first: Optional[bool] = None
    gk: Optional[np.ndarray] = None


class NewtonWolfe(OptimizationMethod):
    max_iters: int
    eps: float
    c1: float = None
    c2: float = None

    def __init__(self, **params):
        self.c1 = params.get("c1", 1e-4)
        self.c2 = params.get("c2", 0.9)
        self.eps = params["aprox_dec"]
        self.learning_rate = params.get("learning_rate", 100)
        self.max_iters = params["max_iters"]

    def initial_step(self, oracul: Oracul, point: np.ndarray, **params) -> WolfeState:
        state = WolfeState(point=point, eps=float('inf'))
        state.epoch_state = oracul.next_state()
        state.prev_point = None
        state.fk = oracul.evaluate(state.point, state.epoch_state, **params)
        state.fj_old = state.fk
        state.alpha_old = 0
        state.first = True
        state.gk = oracul.evaluate_gradient(state.point, state.epoch_state, **params)
        return state

    def step(self, oracul: Oracul, state: WolfeState, **params) -> WolfeState:
        grad = oracul.evaluate_gradient(state.point, state.epoch_state)
        state.eps = np.sqrt(np.dot(grad, grad))
        hess = oracul.evaluate_hessian(state.point, state.epoch_state)
        try:
            inverted = np.linalg.inv(hess)
        except np.linalg.LinAlgError:
            inverted = np.linalg.inv(hess + np.eye(hess.shape[0]) * 1e-11)
        pk = -np.dot(inverted, grad)
        alpha = self.wolfe(oracul, state.point, pk, self.c1, self.c2, 1.0, self.learning_rate, self.max_iters, state)
        state.point = state.point + alpha * pk
        return state

    def meta(self, **params) -> MethodMeta:
        return MethodMeta(name="Wolfe",
                          version=f"({self.c1},{self.c2},eps={self.eps})",
                          description="Wolfes method to finding minimum on the ray ")

    def wolfe(self, oracul: Oracul, x, pk, c1, c2, alpha, alpha_max, max_iters, state):
        proj_gk = np.dot(state.gk, pk)
        fj = oracul.evaluate(x + alpha * pk, state.epoch_state)
        gj = oracul.evaluate_gradient(x + alpha * pk, state.epoch_state)
        proj_gj = np.dot(gj, pk)
        if fj > state.fk + c1 * alpha * proj_gk or not state.first and fj > state.fj_old:
            return self.zoom(state, oracul, state.fj_old, state.alpha_old, alpha, x, state.fk, state.gk, pk, c1, c2,
                             max_iters)
        state.first = False
        if np.fabs(proj_gj) <= c2 * np.fabs(proj_gk):
            return alpha
        if proj_gj >= 0.0:
            return self.zoom(state, oracul, fj, alpha, state.alpha_old, x, state.fk, state.gk, pk, c1, c2,
                             max_iters)
        state.fj_old = fj
        state.alpha_old = alpha
        alpha = min(2.0 * alpha, alpha_max)
        if alpha >= alpha_max:
            return None
        return alpha

    def zoom(self, state: State, oracul: Oracul, f_low, alpha_low, alpha_high, x, fk, gk, pk, c1, c2, max_iters):
        alpha_j = 0
        proj_gk = np.dot(pk, gk)
        for j in range(max_iters):
            alpha_j = 0.5 * (alpha_low + alpha_high)
            fj = oracul.evaluate(x + alpha_j * pk, state.epoch_state)
            if fj > fk + c1 * alpha_j or fj >= f_low:
                alpha_high = alpha_j
                oracul.evaluate_gradient(x + alpha_j * pk, state.epoch_state)
            else:
                gj = oracul.evaluate_gradient(x + alpha_j * pk, state.epoch_state)
                proj_gj = np.dot(gj, pk)
                if np.fabs(proj_gj) <= c2 * np.fabs(proj_gk):
                    return alpha_j
                if proj_gj * (alpha_high - alpha_low) >= 0.0:
                    alpha_high = alpha_j
                alpha_low = alpha_j
                f_low = fj
        return alpha_j
