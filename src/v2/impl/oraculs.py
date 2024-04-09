from inspect import signature
from typing import Callable

import numpy as np
import sympy

from src.v2.model.oracul import Oracul, OraculMeta


class LambdaOracul(Oracul):
    """Class for oracul from lambda"""

    def __init__(self, func: Callable[..., float]) -> None:
        """
        Constructor for lambda oracul
        :param func:        lambda that generate oracul
        """
        self.func = func
        self.dimension = len(signature(func).parameters) + 1

    def evaluate(self, point: np.ndarray, **params) -> float:
        return self.func(*point)

    def get_dimension(self) -> int:
        return len(signature(self.func).parameters) + 1

    def meta(self, **params) -> OraculMeta:
        return OraculMeta(name="LambdaOracul",
                          description="Oracul based on lambda")


class SymbolOracul(Oracul):
    """Oracul that automatically calculates gradients and hessians through sympy"""

    def __init__(self, func, dimensions_order: list[str]):
        self.func = func
        self.dimensions_order = dimensions_order
        self.grad = np.array([])
        for i in self.dimensions_order:
            self.grad = np.append(self.grad, sympy.diff(self.func, i))
        self.hes = []
        for i in self.dimensions_order:
            row_res = []
            for j in self.dimensions_order:
                row_res.append(sympy.diff(sympy.diff(self.func, i), j))
            self.hes.append(np.array(row_res))
        self.hes = np.array(self.hes)

    def map_dim_to_point(self, point: np.ndarray) -> dict:
        return dict(zip(self.dimensions_order, point))

    def evaluate(self, point: np.ndarray, **params) -> float:
        return np.float64(self.func.subs(self.map_dim_to_point(point)))

    def evaluate_gradient(self, point: np.ndarray, **params) -> np.ndarray:
        res = [np.float64(i.subs(self.map_dim_to_point(point))) for i in self.grad]
        return np.array(res, dtype=np.float64)

    def evaluate_hessian(self, point: np.ndarray, **params) -> np.ndarray:
        res = []
        for i in self.hes:
            row_res = []
            for j in i:
                row_res.append(np.float64(j.subs(self.map_dim_to_point(point))))
            res.append(np.array(row_res))
        return np.array(res)

    def get_dimension(self) -> int:
        return len(self.dimensions_order) + 1

    def meta(self, **params) -> OraculMeta:
        return OraculMeta(name="SymbolOracul",
                          description="Oracul based on sympy expression")


class PoweredSumOracul(Oracul):
    """Class for oracul generated by coefficients"""

    def __init__(self, coefficients: list[tuple[float, float]]) -> None:
        """
        Constructor for powered sum oracul
        :param coefficients:    coefficients in style [(coefficient of x, power of x), (coefficient of y, power of y)...]
        """
        self.coefficients = coefficients

    def evaluate(self, point: np.ndarray, **params) -> float:
        res = np.float64(0)
        for coordinate, coefficient, power in zip(point, self.coefficients):
            res += coefficient * (np.float64(coordinate) ** power)
        for i in range(len(self.coefficients)):
            res += self.coefficients[i][0] * (np.float64(point[i]) ** self.coefficients[i][1])
        return res

    def evaluate_gradient(self, point: np.ndarray, **params) -> np.ndarray:
        res = np.zeros(len(self.coefficients), dtype=np.float64)
        for i in range(len(self.coefficients)):
            res[i] = (np.float64(point[i]) ** np.float64(self.coefficients[i][1] - 1)) * \
                     self.coefficients[i][0] * \
                     self.coefficients[i][1]
        return res

    def get_dimension(self) -> int:
        return len(self.coefficients) + 1

    def meta(self, **params) -> OraculMeta:
        return OraculMeta(name="PoweredSumOracul",
                          description="Oracul described by powers and coefficients")
