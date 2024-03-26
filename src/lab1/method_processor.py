import copy
import typing as tp

from matplotlib.animation import Animation

from ..common import Figure
from ..metric.metric import MetricResult
from ..visualization import Animator
from ..metric import Metric, MetricMethod
from ..common import OptimizationMethod, Oracul, Point, State
from .stop_condition import StopCondition


class MethodProcessor:
    @staticmethod
    def process(method: OptimizationMethod, oracul: Oracul, condition: StopCondition,
                metrics: tp.Optional[list[Metric]] = None, visualize: bool = True,
                method_params: tp.Optional[dict[str, tp.Any]] = None,
                metric_params: tp.Optional[dict[str, tp.Any]] = None,
                low_bracket: tp.Optional[list[float]] = None,
                high_bracket: tp.Optional[list[float]] = None,
                **params) -> tuple[tp.Optional[Point], tp.Optional[list[MetricResult]], tp.Optional[Animation]]:
        if method_params is None:
            method_params = {}
        if metric_params is None:
            metric_params = {}
        if metrics is not None:
            method = MetricMethod(method, metrics)

        condition = copy.deepcopy(condition)
        if condition.stop(None, None):
            metric_results, anim = MethodProcessor.stop(method, oracul, [], visualize, metric_params)
            return None, metric_results, anim

        point, state = method.initial_step(oracul, visualize, **method_params)

        points: list[Point] = [point]
        states: list[State] = [state]

        while not condition.stop(point, state):
            point, state = method.step(oracul, state, visualize)
            points.append(point)
            states.append(state)

        coordinates: list[list[float]] = list(map(lambda x: x.coordinates, points))
        if low_bracket is not None:
            coordinates.append(low_bracket + [0])
        if high_bracket is not None:
            coordinates.append(high_bracket + [0])
        low_bracket = [min(i) - 0.1 * abs(min(i)) for i in zip(*coordinates)][:-1]
        high_bracket = [max(i) + 0.1 * abs(max(i)) for i in zip(*coordinates)][:-1]

        metric_results, anim = MethodProcessor.stop(method, oracul, states, visualize, metric_params,
                                                    start=low_bracket, end=high_bracket)
        return point, metric_results, anim

    @staticmethod
    def stop(method: OptimizationMethod, oracul: Oracul, states: list[State], visualize: bool,
             metric_params: dict[str, tp.Any], start: tp.Optional[list[float]] = None,
             end: tp.Optional[list[float]] = None) -> tuple[tp.Optional[list[MetricResult]], Animation]:

        anim: tp.Optional[Animation] = None

        if visualize:
            figures: list[list[Figure]] = list(map(lambda x: x.visual_state, states))
            anim = Animator.animate(figures, oracul, oracul.get_dimension(), step=0.01, start=start, end=end)

        if isinstance(method, MetricMethod):
            return method.get_result(**metric_params), anim
        return None, anim
