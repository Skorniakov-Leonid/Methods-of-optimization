import numpy as np
import typing as tp

from src.common import OptimizationMethod, Point, State, Oracul, PointFigure
from src.common.oracul import GradientOracul, HessianOracul
from src.v2.impl.methods import GradientDescent


# Почему функция знает, визуализируют её или нет
# Прокидка стейта не меет применений
# Learning_Rate хоется подавать через initial_step, наверное - хотя вот хз
# Оборачиваемые оракулы - бред. Новый наследник Oracul = прописывание обработки во всех метриках ненужных им вызовов
class NewtonBase(OptimizationMethod):
    """Newton method. Fixed learning rate"""

    learning_rate: tp.Optional[float] = None
    x: tp.Optional[list[float]] = None
    y: tp.Optional[float] = None
    prev_x: tp.Optional[list[float]] = None

    def get_state(self, visualize: bool = False) -> State:
        if not visualize:
            return State(
                [PointFigure(np.array(self.x).tolist() + [self.y])],
                [],
                self.get_precision()
            )
        return State(
            [PointFigure(np.array(self.x).tolist() + [self.y])],
            [],
            self.get_precision()
        )

    def get_temp_res(self):
        return Point(np.append(np.array(self.x), np.array([self.y])))

    def initial_step(self, oracul: Oracul, visualize: bool = False, **params) -> tuple[Point, State]:
        self.learning_rate = params["learning_rate"]
        self.x = params["start_point"]
        self.y = oracul.evaluate(Point(np.array(self.x, np.float64)))
        self.prev_x = None
        return self.get_temp_res(), self.get_state(visualize)

    def step(self, oracul: HessianOracul, state: State, visualize: bool = False) -> tuple[Point, State]:
        self.prev_x = self.x
        self.x = (np.array(self.x, dtype=np.float64) - self.get_leatning_rate()
                  * np.dot(np.linalg.inv(oracul.evaluate_hessian(Point(np.array(self.x, np.float64)))),
                           oracul.evaluate_gradient(Point(np.array(self.x, np.float64))))
                  )
        self.y = oracul.evaluate(Point(np.array(self.x, dtype=np.float64)))
        return self.get_temp_res(), self.get_state(visualize)

    def name(self) -> str:
        return "Newton with fixed rate"

    def get_leatning_rate(self):
        return self.learning_rate

    def get_precision(self):
        return float("inf") if self.prev_x is None else np.sqrt(np.sum(np.square(self.x - self.prev_x)))


class SteepestDescent(OptimizationMethod):
    c1: float
    c2: float
    max_iters: int
    x: Point
    goracul: GradientOracul
    eps: float

    def initial_step(self, oracul: Oracul, visualize: bool = False, **params) -> tuple[Point, State]:
        self.goracul = params["gradient_oracul"]
        self.c1 = params["c1"]
        self.c2 = params["c2"]
        self.max_iters = params["max_iters"]
        self.x = params["start_point"]
        self.eps = params["eps"]

    def get_state(self, visualize: bool = False) -> State:
        if not visualize:
            return State(
                [PointFigure(np.array(self.x).tolist())],
                [],
                self.eps
            )
        return State(
            [PointFigure(np.array(self.x).tolist())],
            [],
            self.eps
        )

    def step(self, oracul: Oracul, state: State, visualize: bool = False) -> tuple[Point, State]:
        grad = self.goracul.evaluate_gradient(self.x)
        if np.sqrt(np.dot(grad, grad)) < self.eps:
            return self.x, self.get_state()
        pk = -grad / np.sqrt(np.dot(grad, grad))
        alpha = self.wolfe(self.goracul, self.x.coordinates, pk, self.c1, self.c2, 1.0, 100.0, self.max_iters)
        self.x = Point(self.x.coordinates + alpha * pk)
        return self.x, self.get_state(visualize=visualize)

    def name(self) -> str:
        return "Steepest descent"

    def wolfe(self, oracul: GradientOracul, x, pk, c1, c2, alpha, alpha_max, max_iters):
        fk = oracul.evaluate(Point(x))
        gk = oracul.evaluate_gradient(Point(x))
        proj_gk = np.dot(gk, pk)
        fj_old = fk
        proj_gj_old = proj_gk
        alpha_old = 0
        for j in range(max_iters):
            fj = oracul.evaluate(Point(x + alpha * pk))
            gj = oracul.evaluate_gradient(Point(x + alpha * pk))
            proj_gj = np.dot(gj, pk)
            if fj > fk + c1 * alpha * proj_gk or j > 0 and fj > fj_old:
                return self.zoom(oracul, fj_old, proj_gj_old, alpha_old, fj, proj_gj, alpha, x, fk, gk, pk, c1, c2,
                                 max_iters)
            if np.fabs(proj_gj) <= c2 * np.fabs(proj_gk):
                return alpha
            if proj_gj >= 0.0:
                return self.zoom(oracul, fj, proj_gj, alpha, fj_old, proj_gj_old, alpha_old, x, fk, gk, pk, c1, c2,
                                 max_iters)
            fj_old = fj
            proj_gj_old = proj_gj
            alpha_old = alpha
            alpha = min(2.0 * alpha, alpha_max)
            if alpha >= alpha_max:
                return None
        return alpha

    def zoom(self, oracul: GradientOracul, f_low, proj_low, alpha_low, f_high, proj_high, alpha_high, x, fk, gk, pk, c1,
             c2, max_iters):
        alpha_j = 0
        proj_gk = np.dot(pk, gk)
        for j in range(max_iters):
            alpha_j = 0.5 * (alpha_low + alpha_high)
            fj = oracul.evaluate(Point(x + alpha_j * pk))
            if fj > fk + c1 * alpha_j or fj >= f_low:
                alpha_high = alpha_j
                f_high = fj
                gj = oracul.evaluate_gradient(Point(x + alpha_j * pk))
                proj_high = np.dot(gj, pk)
            else:
                gj = oracul.evaluate_gradient(Point(x + alpha_j * pk))
                proj_gj = np.dot(gj, pk)
                if np.fabs(proj_gj) <= c2 * np.fabs(proj_gk):
                    return alpha_j
                if proj_gj * (alpha_high - alpha_low) >= 0.0:
                    alpha_high = alpha_j
                    proj_high = proj_low
                    f_high = f_low
                alpha_low = alpha_j
                proj_low = proj_gj
                f_low = fj
        return alpha_j
