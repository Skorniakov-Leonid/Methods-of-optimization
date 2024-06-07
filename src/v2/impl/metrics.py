import time
import tracemalloc
from multiprocessing import Pipe

import numpy as np
import typing as tp
from typing import Callable
# import resource

import psutil
from memory_profiler import memory_usage

from src.v2.impl.oraculs import MinimisingOracul
# если не заимпортировалось:
# pip install -U memory_profiler

from src.v2.model.meta import Meta
from src.v2.model.method import State
from src.v2.model.metric import MetricModule, MetricMeta
from src.v2.model.oracul import Oracul, OraculMeta, SpyOracul, EpochState


class StepCount(MetricModule):
    def __init__(self) -> None:
        self.step_count = -1

    def process_step(self, state: State, meta: Meta, **params) -> bool:
        self.step_count += 1
        return True

    def meta(self, **params) -> MetricMeta:
        return MetricMeta(name="StepCount",
                          result=self.step_count,
                          description="Count of steps",
                          debug=False)


class CallCount(MetricModule):
    def __init__(self) -> None:
        self.calls_count = 0

    def prepare_oracul(self, oracul: Oracul, **params) -> Oracul:
        class CountingOracul(SpyOracul):
            def evaluate(self_oracul, point: np.ndarray, state: EpochState, **params) -> float:
                self.calls_count += 1
                return self_oracul.oracul.evaluate(point, state, **params)

        return CountingOracul(oracul)

    def meta(self, **params) -> MetricMeta:
        return MetricMeta(name="CallCount",
                          result=self.calls_count,
                          description="Count of oracul's calls",
                          debug=False)


class UniqueCallCount(MetricModule):

    def __init__(self) -> None:
        self.calls_count = 0
        self.calls_from: set[np.ndarray] = set()

    def prepare_oracul(self, oracul: Oracul, **params) -> Oracul:
        class CountingOracul(SpyOracul):
            def evaluate(self_oracul, point: np.ndarray, state: EpochState, **params) -> float:
                if point not in self.calls_from:
                    self.calls_count += 1
                    self.calls_from.add(point)
                return self_oracul.oracul.evaluate(point, state, **params)

        return CountingOracul(oracul)

    def meta(self, **params) -> MetricMeta:
        return MetricMeta(name="UniqueCallCount",
                          result=self.calls_count,
                          description="Count of oracul's unique calls",
                          debug=False)


class GradientCallCount(MetricModule):
    def __init__(self) -> None:
        self.calls_count = 0

    def prepare_oracul(self, oracul: Oracul, **params) -> Oracul:
        class CountingOracul(SpyOracul):
            def evaluate_gradient(self_oracul, point: np.ndarray, state: EpochState, **params) -> np.ndarray:
                self.calls_count += 1
                return self_oracul.oracul.evaluate_gradient(point, state, **params)

        return CountingOracul(oracul)

    def meta(self, **params) -> MetricMeta:
        return MetricMeta(name="GradientCallCount",
                          result=self.calls_count,
                          description="Count of oracul's calls of gradient",
                          debug=False)


class HessianCallCount(MetricModule):
    def __init__(self) -> None:
        self.calls_count = 0

    def prepare_oracul(self, oracul: Oracul, **params) -> Oracul:
        class CountingOracul(SpyOracul):
            def evaluate_hessian(self_oracul, point: np.ndarray, state: EpochState, **params) -> np.ndarray:
                self.calls_count += 1
                return self_oracul.oracul.evaluate_hessian(point, state, **params)

        return CountingOracul(oracul)

    def meta(self, **params) -> MetricMeta:
        return MetricMeta(name="HessianCallCount",
                          result=self.calls_count,
                          description="Count of oracul's calls of hessian",
                          debug=False)


class PrecisionCount(MetricModule):
    def __init__(self, precision: float) -> None:
        self.precision = precision
        self.step_count = 0
        self.success = False

    def process_step(self, state: State, meta: Meta, **params) -> bool:
        self.success = self.success or self.precision > state.eps
        if not self.success:
            self.step_count += 1
        return True

    def meta(self, **params) -> MetricMeta:
        return MetricMeta(name="PrecisionCount",
                          version=f"({self.precision})",
                          result=self.step_count if self.success else "Undefined",
                          description="Count of steps before reaching precision",
                          debug=False)


class AbsolutePrecisionCount(MetricModule):
    def __init__(self, precision: float, point: np.ndarray) -> None:
        self.precision = precision
        self.point = point
        self.step_count = 0
        self.success = False

    def process_step(self, state: State, meta: Meta, **params) -> bool:
        self.success = self.success or self.precision > np.linalg.norm(self.point - state.point)
        if not self.success:
            self.step_count += 1
        return True

    def meta(self, **params) -> MetricMeta:
        return MetricMeta(name="AbsolutePrecisionCount",
                          version=f"({self.precision})",
                          result=self.step_count if self.success else "Undefined",
                          description="Count of steps before reaching absolute precision",
                          debug=False)


class Precision(MetricModule):
    def __init__(self) -> None:
        self.precision = float('inf')

    def process_step(self, state: State, meta: Meta, **params) -> bool:
        self.precision = state.eps
        return True

    def meta(self, **params) -> MetricMeta:
        return MetricMeta(name="Precision",
                          result=self.precision,
                          description="Precision",
                          debug=True,
                          display=False)


class AbsolutePrecision(MetricModule):

    def __init__(self, point: np.ndarray) -> None:
        self.point = point
        self.precision = float('inf')

    def process_step(self, state: State, meta: Meta, **params) -> bool:
        self.precision = np.linalg.norm(self.point - state.point)
        return True

    def meta(self, **params) -> MetricMeta:
        return MetricMeta(name="AbsolutePrecision",
                          result=self.precision,
                          description="Absolute precision",
                          debug=True)


class MinAbsolutePrecision(MetricModule):

    def __init__(self, point: np.ndarray) -> None:
        self.point = point
        self.min_precision = float('inf')

    def process_step(self, state: State, meta: Meta, **params) -> bool:
        self.min_precision = min(self.min_precision, np.linalg.norm(self.point - state.point))
        return True

    def meta(self, **params) -> MetricMeta:
        return MetricMeta(name="MinAbsolutePrecision",
                          result=self.min_precision,
                          description="Minimal absolute precision",
                          debug=True)


class EpochCount(MetricModule):
    def __init__(self) -> None:
        self.epoch_count = 0

    def process_step(self, state: State, meta: Meta, **params) -> bool:
        self.epoch_count = state.epoch_state.epoch
        return True

    def meta(self, **params) -> MetricMeta:
        return MetricMeta(name="EpochCount",
                          result=self.epoch_count,
                          description="Count of epochs",
                          debug=True)


class ExecutionTime(MetricModule):
    def __init__(self):
        self.total_time = 0.0

    def prepare_method(self, method: Callable, **params) -> Callable:
        def timedMethod(*args, **params):
            start_time = time.time()

            result = method(*args, **params)

            self.total_time += time.time() - start_time

            return result

        return timedMethod

    def meta(self, **params) -> MetricMeta:
        return MetricMeta(name="ExecutionTime",
                          result=self.total_time,
                          description="Time of execution",
                          debug=True)


class RAMSize(MetricModule):
    def __init__(self):
        self.mem = 0

    def prepare_method(self, method: Callable, **params) -> Callable:
        def ramedMethod(*args, **params):
            start_mem = memory_usage(max_usage=True)
            mem, result = memory_usage((method, args, params), interval=0.01, max_iterations=1, retval=True,
                                       max_usage=True, backend="psutil")

            self.mem = mem - start_mem
            return result

        return ramedMethod

    def meta(self, **params) -> MetricMeta:
        return MetricMeta(name="RAMSize",
                          result=str(self.mem) + " Mb",
                          description="Busy ram",
                          debug=False)


class OraculValue(MetricModule):
    def __init__(self):
        self.value: tp.Union[str, float] = "Undefined"

    def prepare_oracul(self, oracul: Oracul, **params) -> Oracul:
        class ValueOracul(SpyOracul):
            def evaluate(self_oracul, point: np.ndarray, state: EpochState, **params) -> float:
                self.value = self_oracul.oracul.evaluate(point, state, **params)
                return self.value

        return ValueOracul(oracul, **params)

    def meta(self, **params) -> MetricMeta:
        return MetricMeta(name="OraculValue",
                          result=self.value,
                          description="Value of oracul",
                          debug=True)


class LossValue(MetricModule):
    def __init__(self):
        self.oracul: tp.Optional[Oracul] = None
        self.last_point: tp.Optional[np.ndarray] = None

    def prepare_oracul(self, oracul: Oracul, **params) -> Oracul:
        if oracul.get_data() is not None:
            self.oracul = oracul
        return oracul

    def process_step(self, state: State, meta: Meta, **params) -> bool:
        self.last_point = state.point
        return True

    def meta(self, **params) -> MetricMeta:
        return MetricMeta(name="LossValue",
                          result="Undefined" if self.oracul is None or self.last_point is None
                          else self.oracul.evaluate(
                              self.last_point,
                              EpochState(
                                  points=list(
                                      range(
                                          len(self.oracul.get_data())
                                      )
                                  )
                              )
                          ),
                          description="Value of loss function",
                          debug=True)


class ResultValue(MetricModule):
    def __init__(self):
        self.point: tp.Union[str, float] = "Undefined"

    def process_step(self, state: State, meta: Meta, **params) -> bool:
        self.point = state.point
        # print(self.point)
        return True

    def meta(self, **params) -> MetricMeta:
        return MetricMeta(name="Result",
                          result=self.point,
                          description="Result point",
                          debug=True)
