import copy
from typing import Any
from tabulate import tabulate
import numpy as np

from src.v2.model.method import OptimizationMethod
from src.v2.model.metric import MetricModule
from src.v2.model.oracul import Oracul
from src.v2.runner.debug import DebugMetricModule, DebugOracul, DebugMethod
from src.v2.runner.pipeline import PipelineModule, Pipeline
from src.v2.visualization.visualization import VisualizationModule

VISUALIZE = {
    "visualize": True,
}

FULL_VISUALIZE = {"animate": True,
                  "animate_main": True,
                  "animate_contour": True
                  }

FULL_ANIMATION = {"animation_main_full": True,
                  "animation_contour_full": True
                  }

VISUALIZE_FUNCTION = {"visualize_function": True}

NO_VISUALIZE = {"animate": False,
                "visualize": False}

TABLE = {"show_table": True}


class PipelineBuilder:
    @staticmethod
    def build(modules: list[PipelineModule], **params) -> Pipeline:
        debug_metric = params.get("debug_metric")
        visualize = params.get("visualize")

        prepared_modules: list[PipelineModule] = []
        for module in modules:
            module = copy.deepcopy(module)
            if debug_metric and isinstance(module, MetricModule):
                prepared_modules += [DebugMetricModule(module)]
            elif isinstance(module, VisualizationModule):
                if visualize:
                    prepared_modules += [module]
            else:
                prepared_modules += [module]

        return Pipeline(prepared_modules)


class Runner:
    @staticmethod
    def run(methods: list[OptimizationMethod], oraculs: list[Oracul], point: np.ndarray,
            modules: list[PipelineModule], **params) -> list[list[list[tuple[str, Any]]]]:
        data = [[Runner.run_pipeline(method, oracul, point, modules, **params) for method in methods]
                for oracul in oraculs]
        return [Runner.info(oracul, table, **params) for oracul, table in zip(oraculs, data)]

    @staticmethod
    def run_methods(methods: list[OptimizationMethod], oracul: Oracul, point: np.ndarray,
                    modules: list[PipelineModule], **params) -> list[list[tuple[str, Any]]]:
        return Runner.info(oracul,
                           [Runner.run_pipeline(method, oracul, point, modules, **params) for method in methods],
                           **params)

    @staticmethod
    def run_method(method: OptimizationMethod, oracul: Oracul, point: np.ndarray,
                   modules: list[PipelineModule], **params) -> list[list[tuple[str, Any]]]:
        return Runner.info(oracul, [Runner.run_pipeline(method, oracul, point, modules, **params)], **params)

    @staticmethod
    def run_pipeline(method: OptimizationMethod, oracul: Oracul, point: np.ndarray,
                     modules: list[PipelineModule], **params) -> list[tuple[str, Any]]:
        params["method_name"] = method.meta().full_name()
        info = params.get("info", False)
        if info:
            print(f"============ Testing {method.meta().full_name()} ============")
        if params.get("debug_method"):
            method = DebugMethod(method)

        pipeline = PipelineBuilder.build(modules, **params)

        prepared_oracul = pipeline.prepare_oracul(oracul, **params)
        if params.get("debug_oracul"):
            prepared_oracul = DebugOracul(prepared_oracul)

        def run_method():
            state = method.initial_step(prepared_oracul, copy.deepcopy(point), **params)
            while pipeline.process_step(state, method.meta(), **params):
                state.index += 1
                state.epoch_state = oracul.next_state(state.epoch_state, **params)
                state = method.step(prepared_oracul, state, **params)
            return state

        state = pipeline.prepare_method(run_method)()

        if info:
            print(f"[INFO][Method][{method.meta().full_name()}] Completed!")
        return ([("no_show_value", state.point), ("Method name", method.meta().full_name())]
                + pipeline.get_result(**params))

    @staticmethod
    def info(oracul: Oracul, data: list[list[tuple[str, Any]]], **params) -> list[list[tuple[str, Any]]]:
        if len(data) != 0 and params.get("show_table"):
            showing_data = [[tpl for tpl in row if not tpl[0].startswith("no_show")] for row in data]
            headers = [tpl[0] for tpl in showing_data[0]]
            table = [[tpl[1] for tpl in row] for row in showing_data]
            string_table = tabulate(table, headers=headers, tablefmt="grid")
            print(f"\n{oracul.meta().full_name()}\n{string_table}")
        return data
