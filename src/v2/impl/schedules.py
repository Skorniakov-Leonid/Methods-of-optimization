from src.v2.model.method import State
from src.v2.model.schedule import schedule


def epoch_schedule(learning_rate: float, decay: float, clazz):
    def epoch_scheduler(base_rate: float, state: State) -> float:
        return base_rate / (1 + decay * state.epoch_state.epoch)

    return schedule(learning_rate, epoch_scheduler, clazz)


def step_schedule(learning_rate: float, decay: float, clazz):
    def step_scheduler(base_rate: float, state: State) -> float:
        return base_rate / (1 + decay * state.index)

    return schedule(learning_rate, step_scheduler, clazz)


def pow_step_schedule(learning_rate: float, decay: float, pow: float, clazz):
    def step_scheduler(base_rate: float, state: State) -> float:
        return base_rate / (1 + decay * state.index ** pow)

    return schedule(learning_rate, step_scheduler, clazz)


def ladder_schedule(learning_rate: float, clazz):
    def ladder_scheduler(base_rate: float, state: State) -> float:
        if state.epoch_state.epoch >= 10:
            return base_rate / 10
        if state.epoch_state.epoch >= 20:
            return base_rate / 100
        return base_rate

    return schedule(learning_rate, ladder_scheduler, clazz)
