"""
Steps:
    - Load data
    - Load pipeline/model
    - Run inference on all test data
        - Save outputs of specified model (on clean data)
        - Calculate error of specified model (on clean data)

    - Save results
        - File structure:
            - output/check_callibration/pipeline_name/run_[X]/[OUTPUT FILES HERE]
"""

import argparse
import json
import os

from arc_spice.data.multieurlex_utils import load_multieurlex
from arc_spice.eval.inference_utils import ResultsGetter, run_inference
from arc_spice.utils import open_yaml_path
from arc_spice.variational_pipelines.RTC_single_component_pipeline import (
    RTCSingleComponentPipeline,
)

OUTPUT_DIR = "outputs"


def main(args):
    # initialise pipeline
    data_config = open_yaml_path(args.data_config)
    pipeline_config = open_yaml_path(args.pipeline_config)
    (_, test_loader, _), meta_data = load_multieurlex(**data_config)
    rtc_single_component_pipeline = RTCSingleComponentPipeline(
        model_pars=pipeline_config, data_pars=meta_data, model_key=args.model_key
    )
    results_getter = ResultsGetter(meta_data["n_classes"])

    test_results = run_inference(
        dataloader=test_loader,
        pipeline=rtc_single_component_pipeline,
        results_getter=results_getter,
    )

    data_name = args.data_config.split("/")[-1].split(".")[0]
    pipeline_name = args.pipeline_config.split("/")[-1].split(".")[0]
    save_loc = (
        f"{OUTPUT_DIR}/inference_results/{data_name}/{pipeline_name}/"
        f"single_component"
    )
    os.makedirs(save_loc, exist_ok=True)

    with open(f"{save_loc}/{args.model_key}.json", "w") as save_file:
        json.dump(test_results, save_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "From an experiment path generates evaluation plots for every experiment."
        )
    )
    parser.add_argument(
        "pipeline_config",
        type=str,
        default=None,
        help="Path to pipeline config.",
    )
    parser.add_argument(
        "data_config",
        type=str,
        default=None,
        help="Path to data config.",
    )

    parser.add_argument(
        "model_key",
        type=str,
        default=None,
        help="Model on which to run inference.",
    )

    args = parser.parse_args()

    main(args)
