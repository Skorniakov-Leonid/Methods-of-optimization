from typing import Callable, Any

from src.v2.model.method import State, MethodMeta


def schedule(learning_rate: float, schedule_function: Callable[..., float], clazz, **params):
    class ScheduleMethod(clazz):
        def get_learning_rate(self, state: State, **args) -> float:
            return schedule_function(learning_rate, state, **args)

        def meta(self, **args) -> MethodMeta:
            meta = super().meta(**args)
            return MethodMeta(name=f"Schedule|{meta.name}",
                              version=meta.version,
                              description=meta.description)

    return ScheduleMethod
