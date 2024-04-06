import numpy as np
import typing as tp

from src.common import OptimizationMethod, Point, State, Oracul, PointFigure
from src.common.oracul import GradientOracul, HessianOracul


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
