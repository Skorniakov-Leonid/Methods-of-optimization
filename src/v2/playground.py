import matplotlib.pyplot as plt
import numpy as np

from src.v2.impl.conditions import StepCountCondition, PrecisionCondition, AbsolutePrecisionCondition
from src.v2.impl.methods import CoordinateDescent, GoldenRatioMethod, GradientDescent, ScipyMethod
from src.v2.impl.metrics import StepCount, CallCount, GradientCallCount, HessianCallCount, PrecisionCount, \
    AbsolutePrecisionCount, MinAbsolutePrecision, AbsolutePrecision, Precision
from src.v2.impl.oraculs import LambdaOracul
from src.v2.runner.debug import FULL_DEBUG
from src.v2.runner.runner import Runner, TABLE, FULL_VISUALIZE
from src.v2.visualization.animation import Animator

def test():
    lmb = lambda x, y: (x - 10) ** 2 + (y - 5) ** 2
    oracul = LambdaOracul(lmb)

    min_point = np.array([10, 5])

    animations = [Animator()]
    metrics = [StepCount(), CallCount(), GradientCallCount(), HessianCallCount(), Precision(), PrecisionCount(1),
               AbsolutePrecisionCount(0.001, min_point), AbsolutePrecision(min_point),
               MinAbsolutePrecision(min_point)]
    conditions = [StepCountCondition(50), PrecisionCondition(0.0001),
                  AbsolutePrecisionCondition(0.0000001, min_point)]

    modules = animations + metrics + conditions

    methods = [  # GoldenRatioMethod(),
        CoordinateDescent(),
        # GradientDescent(),
        # ScipyMethod("Newton-CG"),
        # ScipyMethod("Nelder-Mead"),
        # ScipyMethod("BFGS")
    ]
    oraculs = [oracul]
    point = np.array([20, 20])

    result = Runner.run(methods, oraculs, point, modules, info=True, **FULL_DEBUG, **FULL_VISUALIZE, **TABLE, precision=1e-7)

    plt.show()


if __name__ == "__main__":
    test()
