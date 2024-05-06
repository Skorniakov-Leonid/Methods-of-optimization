from abc import abstractmethod, ABC
from typing import Optional, Any, Type

import numpy as np

from v2.model.meta import Meta
from v2.model.method import State
from v2.model.oracul import Oracul


class PipelineModule(ABC):
    def prepare_oracul(self, oracul: Oracul, **params) -> Oracul:
        return oracul

    def process_step(self, state: State, meta: Meta, **params) -> bool:
        return True

    def get_result(self, **params) -> list[tuple[str, Any]]:
        return []

    @abstractmethod
    def meta(self, **params) -> Meta:
        pass


class Pipeline:
    def __init__(self, modules: list[PipelineModule]) -> None:
        self.modules = modules

    def prepare_oracul(self, oracul: Oracul, **params) -> Oracul:
        for module in self.modules:
            oracul = module.prepare_oracul(oracul, **params)
        return oracul

    def process_step(self, state: State, meta: Meta, **params) -> bool:
        stop = True
        for module in self.modules:
            stop = stop and module.process_step(state, meta, **params)
        return stop

    def get_result(self, **params) -> list[tuple[str, Any]]:
        result: list[tuple[str, Any]] = []
        for module in self.modules:
            tpl = module.get_result(**params)
            result += tpl
        return result
