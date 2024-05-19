from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import typing as tp
import numdifftools as nd

from src.v2.model.function_interpretation import FunctionInterpretation
from src.v2.model.meta import Meta


@dataclass
class EpochState:
    epoch: int = 0
    step: int = 0
    points: list[int] = field(default_factory=lambda: [])
    previous_points: list[int] = field(default_factory=lambda: [])


@dataclass
class OraculMeta(Meta):
    """Class container for meta information about oracul"""
    pass


class Oracul(ABC):
    """Interface for oracul (anonymous value generator in point)"""

    @abstractmethod
    def evaluate(self, point: np.ndarray, state: EpochState, **params) -> float:
        """
        Get value in point
        :param point:       point for evaluating
        :param state:       epoch state
        :return:            calculated value
        """
        pass

    def evaluate_gradient(self, point: np.ndarray, state: EpochState, **params) -> np.ndarray:
        """
        Get gradient in point
        :param point:       point for calculating gradient
        :param state:       epoch state
        :return:            calculated gradient
        """
        return nd.Gradient(partial(self.evaluate, state=state))(point)

    def evaluate_hessian(self, point: np.ndarray, state: EpochState, **params) -> np.ndarray:
        """
        Get hessian in point
        :param point:       point for calculating gradient
        :param state:       epoch state
        :return:            calculated hessian
        """
        return nd.Hessian(partial(self.evaluate, state=state))(point)

    def next_state(self, state: EpochState = None, **params) -> EpochState:
        if state is None:
            return EpochState(0, 0, np.empty(0))
        return EpochState(state.epoch + 1, 0, np.empty(0))

    def all_points(self, **params) -> EpochState:
        return EpochState(-1, -1, [], [])

    def get_interpretation(self) -> tp.Optional[FunctionInterpretation]:
        return None

    def get_data(self) -> tp.Optional[np.ndarray]:
        return None

    @abstractmethod
    def get_dimension(self) -> int:
        pass

    @abstractmethod
    def meta(self, **params) -> OraculMeta:
        pass


class SpyOracul(Oracul):
    def __init__(self, oracul: Oracul, **params) -> None:
        self.oracul = oracul

    def evaluate(self, point: np.ndarray, state: EpochState, **params) -> float:
        return self.oracul.evaluate(point, state, **params)

    def evaluate_gradient(self, point: np.ndarray, state: EpochState, **params) -> np.ndarray:
        return self.oracul.evaluate_gradient(point, state, **params)

    def evaluate_hessian(self, point: np.ndarray, state: EpochState, **params) -> np.ndarray:
        return self.oracul.evaluate_hessian(point, state, **params)

    def next_state(self, state: EpochState = None, **params) -> EpochState:
        return self.oracul.next_state(state, **params)

    def all_points(self, **params) -> EpochState:
        return self.oracul.all_points()

    def get_interpretation(self) -> tp.Optional[FunctionInterpretation]:
        return self.oracul.get_interpretation()

    def get_data(self) -> tp.Optional[np.ndarray]:
        return self.oracul.get_data()

    def get_dimension(self) -> int:
        return self.oracul.get_dimension()

    def meta(self, **params) -> OraculMeta:
        return self.oracul.meta(**params)
