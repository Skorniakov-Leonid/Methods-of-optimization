import numpy as np

from src.v2.model.meta import Meta
from src.v2.model.method import State
from src.v2.model.metric import MetricModule, MetricMeta
from src.v2.model.oracul import Oracul, OraculMeta


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
        class CountingOracul(Oracul):
            def evaluate(self_oracul, point: np.ndarray, **params) -> float:
                self.calls_count += 1
                return oracul.evaluate(point, **params)

            def evaluate_gradient(self_oracul, point: np.ndarray, **params) -> np.ndarray:
                return oracul.evaluate_gradient(point, **params)

            def get_dimension(self_oracul) -> int:
                return oracul.get_dimension()

            def evaluate_hessian(self, point: np.ndarray, **params) -> np.ndarray:
                return oracul.evaluate_hessian(point, **params)

            def meta(self, **params) -> OraculMeta:
                return oracul.meta()

        return CountingOracul()

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
        class CountingOracul(Oracul):
            def evaluate(self_oracul, point: np.ndarray, **params) -> float:
                if point not in self.calls_from:
                    self.calls_count += 1
                    self.calls_from.add(point)
                return oracul.evaluate(point, **params)

            def evaluate_gradient(self_oracul, point: np.ndarray, **params) -> np.ndarray:
                return oracul.evaluate_gradient(point, **params)

            def get_dimension(self_oracul) -> int:
                return oracul.get_dimension()

            def evaluate_hessian(self, point: np.ndarray, **params) -> np.ndarray:
                return oracul.evaluate_hessian(point, **params)

            def meta(self, **params) -> OraculMeta:
                return oracul.meta()

        return CountingOracul()

    def meta(self, **params) -> MetricMeta:
        return MetricMeta(name="UniqueCallCount",
                          result=self.calls_count,
                          description="Count of oracul's unique calls",
                          debug=False)


class GradientCallCount(MetricModule):
    def __init__(self) -> None:
        self.calls_count = 0

    def prepare_oracul(self, oracul: Oracul, **params) -> Oracul:
        class CountingOracul(Oracul):
            def evaluate(self_oracul, point: np.ndarray, **params) -> float:
                return oracul.evaluate(point, **params)

            def evaluate_gradient(self_oracul, point: np.ndarray, **params) -> np.ndarray:
                self.calls_count += 1
                return oracul.evaluate_gradient(point, **params)

            def get_dimension(self_oracul) -> int:
                return oracul.get_dimension()

            def evaluate_hessian(self, point: np.ndarray, **params) -> np.ndarray:
                return oracul.evaluate_hessian(point, **params)

            def meta(self, **params) -> OraculMeta:
                return oracul.meta()

        return CountingOracul()

    def meta(self, **params) -> MetricMeta:
        return MetricMeta(name="GradientCallCount",
                          result=self.calls_count,
                          description="Count of oracul's calls of gradient",
                          debug=False)


class HessianCallCount(MetricModule):
    def __init__(self) -> None:
        self.calls_count = 0

    def prepare_oracul(self, oracul: Oracul, **params) -> Oracul:
        class CountingOracul(Oracul):
            def evaluate(self_oracul, point: np.ndarray, **params) -> float:
                return oracul.evaluate(point, **params)

            def evaluate_gradient(self_oracul, point: np.ndarray, **params) -> np.ndarray:
                return oracul.evaluate_gradient(point, **params)

            def get_dimension(self_oracul) -> int:
                return oracul.get_dimension()

            def evaluate_hessian(self_oracul, point: np.ndarray, **params) -> np.ndarray:
                self.calls_count += 1
                return oracul.evaluate_hessian(point, **params)

            def meta(self, **params) -> OraculMeta:
                return oracul.meta()

        return CountingOracul()

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
