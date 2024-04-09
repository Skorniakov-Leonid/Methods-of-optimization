import matplotlib.pyplot as plt
import numpy as np

from src.v2.impl.conditions import StepCountCondition, PrecisionCondition, AbsolutePrecisionCondition
from src.v2.impl.methods import CoordinateDescent, GoldenRatioMethod, GradientDescent
from src.v2.impl.metrics import StepCount, CallCount, GradientCallCount, HessianCallCount, PrecisionCount, \
    AbsolutePrecisionCount, MinAbsolutePrecision, AbsolutePrecision
from src.v2.impl.oraculs import LambdaOracul
from src.v2.runner.debug import FULL_DEBUG
from src.v2.runner.runner import Runner, TABLE
from src.v2.visualization.animation import Animator, FULL_ANIMATION


# Этот файл нужно удалить

def test():
    lmb = lambda x, y: (x - 10) ** 2 + (y - 5) ** 2
    oracul = LambdaOracul(lmb)

    min_point = np.array([10, 5])

    animations = [Animator()]
    metrics = [StepCount(), CallCount(), GradientCallCount(), HessianCallCount(), PrecisionCount(1),
               AbsolutePrecisionCount(0.001, min_point), AbsolutePrecision(min_point),
               MinAbsolutePrecision(min_point)]
    conditions = [StepCountCondition(50), PrecisionCondition(0.0001),
                  AbsolutePrecisionCondition(0.001, min_point)]

    modules = animations + metrics + conditions

    methods = [#GoldenRatioMethod(),
               CoordinateDescent(),
                GradientDescent()]
    oraculs = [oracul]
    point = np.array([126, 30])

    result = Runner.run(methods, oraculs, point, modules, **TABLE)

    plt.show()


if __name__ == "__main__":
    test()
