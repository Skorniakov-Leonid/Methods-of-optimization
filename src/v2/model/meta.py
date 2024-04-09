from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional


@dataclass
class Meta(ABC):
    """Class for meta information"""
    name: str
    description: str
    type: Optional[str] = None
    version: Optional[str] = None

    def full_name(self, **params) -> str:
        """Get full name from meta information"""
        tokens = [self.name, self.type, self.version]
        return ''.join([token for token in tokens if token is not None])
