import numpy as np

from v2.model.condition import StopConditionModule, StopConditionMeta
from v2.model.meta import Meta
from v2.model.method import State


class StepCountCondition(StopConditionModule):
    def __init__(self, max_count: int) -> None:
        self.max_count = max_count
        self.count = 0

    def process_step(self, state: State, meta: Meta, **params) -> bool:
        self.count += 1
        cond = self.max_count > self.count
        info = params.get("info", False)
        if not cond and info:
            print(f"[INFO][Condition][{self.meta().full_name()}] Condition reached!")
        return cond

    def meta(self, **params) -> StopConditionMeta:
        return StopConditionMeta(name="StepCountCondition",
                                 version=f"(max_count={self.max_count})",
                                 description="Stop when the required number of steps is reached")


class PrecisionCondition(StopConditionModule):
    def __init__(self, precision: float) -> None:
        self.precision = precision

    def process_step(self, state: State, meta: Meta, **params) -> bool:
        cond = self.precision < state.eps
        info = params.get("info", False)
        if not cond and info:
            print(f"[INFO][Condition][{self.meta().full_name()}] Condition reached!")
        return cond

    def meta(self, **params) -> StopConditionMeta:
        return StopConditionMeta(name="PrecisionCondition",
                                 version=f"(precision={self.precision})",
                                 description="Stop when the required precision is reached")


class AbsolutePrecisionCondition(StopConditionModule):
    def __init__(self, precision: float, point: np.ndarray) -> None:
        self.precision = precision
        self.point = point

    def process_step(self, state: State, meta: Meta, **params) -> bool:
        cond = self.precision < np.linalg.norm(self.point - state.point)
        info = params.get("info", False)
        if not cond and info:
            print(f"[INFO][Condition][{self.meta().full_name()}] Condition reached!")
        return cond

    def meta(self, **params) -> StopConditionMeta:
        return StopConditionMeta(name="AbsolutePrecisionCondition",
                                 version=f"(precision={self.precision})",
                                 description="Stop when the required absolute precision is reached")
