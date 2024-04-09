from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Any, Union

from src.v2.model.meta import Meta
from src.v2.runner.pipeline import PipelineModule


@dataclass
class MetricMeta(Meta):
    """Class container for meta information about metric"""
    result: Union[float, str] = "Undefined"
    debug: bool = True
    display: bool = True


class MetricModule(PipelineModule):
    def get_result(self, **params) -> list[tuple[str, Any]]:
        meta = self.meta(**params)
        return [(meta.full_name(), meta.result)]

    @abstractmethod
    def meta(self, **params) -> MetricMeta:
        pass
