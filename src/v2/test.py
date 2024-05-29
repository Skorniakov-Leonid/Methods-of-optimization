from math import sin
from functools import partial
# %matplotlib
# widget
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import sympy
from src.v2.impl.conditions import StepCountCondition, PrecisionCondition, AbsolutePrecisionCondition, \
    EpochCountCondition
from src.v2.impl.function_interpretations import LinearInterpretation, PolynomialInterpretation, LambdaInterpretation
from src.v2.impl.loss_functions import MSE, LineBinary, L1
from src.v2.impl.methods import CoordinateDescent, GoldenRatioMethod, NewtonWolfe, GradientDescentBase
from src.v2.impl.metrics import StepCount, CallCount, GradientCallCount, HessianCallCount, PrecisionCount, \
    AbsolutePrecisionCount, AbsolutePrecision, MinAbsolutePrecision, RAMSize, ExecutionTime, EpochCount, \
    OraculValue, LossValue, ResultValue
from src.v2.impl.oraculs import LambdaOracul, SymbolOracul, MinimisingOracul
from src.v2.runner.debug import FULL_DEBUG
from src.v2.runner.runner import Runner, FULL_VISUALIZE, NO_VISUALIZE, FULL_ANIMATION, VISUALIZE, VISUALIZE_FUNCTION
from src.v2.visualization.animation import Animator
from src.v2.runner.runner import TABLE
from src.v2.impl.methods import GradientDescent, ScipyMethod, Newton, NewtonBase
from IPython.display import display, HTML

# display(HTML("<style>pre { white-space: pre !important; }</style>"))


def print_points(data):
    for i in data:
        print(i[0] + i[1])


metrics_base = [CallCount(), GradientCallCount(), LossValue()]

minPrec = 1e-5
defPrec = 1e-9
maxPrec = 1e-11

animations = [Animator()]

conditions = [StepCountCondition(100)]

def noised_data_generator(func, num, begin, end, noise_range):
    res = []
    for i in range(0, num):
        point = random.uniform(begin, end)
        res += [[point, func([point])[-1] + random.uniform(-noise_range, noise_range)]]
    return res


point_count = 30
crop_size = point_count
noise = 0
diap = 5


def waveSum(
        a2: float, a3: float,
        # b2: float, b3: float,
        # c1: float, c2: float, c3: float,
        point: np.ndarray) -> np.ndarray:
    x = point[0]
    wave1 = sin(a2 * x)

    # wave2 = sin(b2 * x)
    # wave3 = c1 * sin(c2 * x + c3)
    wave = (0
            + wave1
            # + wave2
            # + wave3
            )

    return np.array([x, wave])


coefficients = [
    1, 1
]
wave = partial(waveSum, *coefficients)

data = noised_data_generator(wave, point_count, -diap, diap, noise)
wave_oracul = MinimisingOracul(MSE(), LambdaInterpretation(func=waveSum, eval_dim=2), data, crop_size)

conditions = [EpochCountCondition(1)]
modules = ([ExecutionTime()]
           # + [RAMSize()]
           + [EpochCount()]
           + metrics_base
           + conditions
           + [Animator()]
           + [ResultValue()]
           + [ResultValue()]
           )

methods = [GradientDescent(learning_rate=100, aprox_dec=1e-5)]

oraculs = [
    wave_oracul,
    # MinimisingOracul(L1(MSE(), 0.01), LambdaInterpretation(func = waveSum, eval_dim = 2), data, crop_size),
]

point = np.array([1, 1])


if __name__ == "__main__":
    result = Runner.run(methods,
                        oraculs,
                        point,
                        modules, precision=defPrec,
                        # **FULL_DEBUG,
                        **TABLE,
                        **VISUALIZE,
                        # **FULL_VISUALIZE,
                        # **FULL_ANIMATION,
                        **VISUALIZE_FUNCTION
                        )
