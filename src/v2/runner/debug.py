from typing import Optional, Any, Callable

import numpy as np
import typing as tp

from src.v2.model.function_interpretation import FunctionInterpretation
from src.v2.model.meta import Meta
from src.v2.model.method import State, OptimizationMethod, MethodMeta
from src.v2.model.metric import MetricModule, MetricMeta
from src.v2.model.oracul import Oracul, OraculMeta, EpochState, SpyOracul
from src.v2.runner.pipeline import PipelineModule

FULL_DEBUG = {'debug_oracul': True,
              'debug_metric': True,
              'debug_method': True}


class DebugMethod(OptimizationMethod):
    def __init__(self, method: OptimizationMethod) -> None:
        self.method = method

    def initial_step(self, oracul: Oracul, point: np.ndarray, **params) -> State:
        meta = self.method.meta()
        state = self.method.initial_step(oracul, point, **params)

        print(f"[DEBUG][Method][{meta.full_name()}] Initial state is {state}")

        return state

    def step(self, oracul: Oracul, state: State, **params) -> State:
        state = self.method.step(oracul, state, **params)

        meta = self.method.meta()
        print(f"[DEBUG][Method][{meta.full_name()}][{state.index}][State] {state}")

        return state

    def meta(self, **params) -> MethodMeta:
        return self.method.meta()


class DebugMetricModule(MetricModule):
    def __init__(self, metric: MetricModule) -> None:
        self.metric = metric

    def prepare_oracul(self, oracul: Oracul, **params) -> Oracul:
        prepared_oracul = self.metric.prepare_oracul(oracul, **params)

        meta = self.metric.meta(**params)
        if meta.debug:
            print(f"[DEBUG][Metric][{meta.full_name()}] Prepared oracul")

        return prepared_oracul

    def process_step(self, state: State, meta: Meta, **params) -> bool:
        return_value = self.metric.process_step(state, meta, **params)

        meta = self.metric.meta(**params)
        if meta.debug:
            metric_result = self.metric.get_result(**params)
            print(f"[DEBUG][Metric][{meta.full_name()}][{state.index}] {metric_result[0][1]}")

        return return_value

    def prepare_method(self, method: Callable, **params) -> Callable:
        return self.metric.prepare_method(method)

    def get_result(self, **params) -> list[tuple[str, Any]]:
        metric_result = self.metric.get_result(**params)

        meta = self.metric.meta(**params)
        if meta.debug:
            print(f"[DEBUG][Metric][{meta.full_name()}][Final] {metric_result[0][1]}")

        return metric_result

    def meta(self, **params) -> MetricMeta:
        return self.metric.meta()


class DebugOracul(SpyOracul):
    def __init__(self, oracul: Oracul) -> None:
        self.oracul = oracul
        print("[DEBUG][Oracul] Added debug to oracul")

    def evaluate(self, point: np.ndarray, state: EpochState, **params) -> float:
        oracul_result = self.oracul.evaluate(point, state)

        print(f"[DEBUG][Oracul][Value] Evaluated at {point} - {oracul_result}")

        return oracul_result

    def evaluate_gradient(self, point: np.ndarray, state: EpochState, **params) -> np.ndarray:
        oracul_result = self.oracul.evaluate_gradient(point, state)

        print(f"[DEBUG][Oracul][Gradient] Evaluated at {point} - {oracul_result}")

        return oracul_result

    def evaluate_hessian(self, point: np.ndarray, state: EpochState, **params) -> np.ndarray:
        oracul_result = self.oracul.evaluate_hessian(point, state)

        print(f"[DEBUG][Oracul][Hessian] Evaluated at {point} - {oracul_result}")

        return oracul_result

    def next_state(self, state: EpochState = None, **params) -> EpochState:
        next_state = self.oracul.next_state(state, **params)

        print(f"[DEBUG][Oracul][State] {next_state.epoch} {next_state.points}")

        return next_state

    def get_dimension(self) -> int:
        return self.oracul.get_dimension()

    def meta(self, **params) -> OraculMeta:
        return self.oracul.meta()
