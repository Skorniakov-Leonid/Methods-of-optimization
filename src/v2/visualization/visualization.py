from abc import abstractmethod
from dataclasses import dataclass

from src.v2.model.meta import Meta
from src.v2.runner.pipeline import PipelineModule


@dataclass
class VisualizationMeta(Meta):
    """Class container for meta information about visualization"""
    pass


class VisualizationModule(PipelineModule):
    @abstractmethod
    def meta(self) -> VisualizationMeta:
        pass