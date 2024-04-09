from abc import abstractmethod
from dataclasses import dataclass

import numpy as np

from src.v2.model.meta import Meta
from src.v2.model.method import State
from src.v2.runner.pipeline import PipelineModule


@dataclass
class StopConditionMeta(Meta):
    """Class container for meta information about stop condition"""
    pass


class StopConditionModule(PipelineModule):
    @abstractmethod
    def process_step(self, state: State, meta: Meta, **params) -> bool:
        pass

    @abstractmethod
    def meta(self, **params) -> StopConditionMeta:
        pass
