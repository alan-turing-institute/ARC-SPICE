import argparse
import json
import os

from arc_spice.data.multieurlex_utils import load_multieurlex_for_translation
from arc_spice.eval.inference_utils import ResultsGetter, run_inference
from arc_spice.utils import open_yaml_path
from arc_spice.variational_pipelines.RTC_variational_pipeline import (
    RTCVariationalPipeline,
)

OUTPUT_DIR = "outputs"


def main(args):
    # initialise pipeline
    data_config = open_yaml_path(args.data_config)
    pipeline_config = open_yaml_path(args.pipeline_config)
    data_sets, meta_data = load_multieurlex_for_translation(**data_config)
    test_loader = data_sets["test"]
    rtc_variational_pipeline = RTCVariationalPipeline(
        model_pars=pipeline_config, data_pars=meta_data
    )
    results_getter = ResultsGetter(meta_data["n_classes"])

    test_results = run_inference(
        dataloader=test_loader,
        pipeline=rtc_variational_pipeline,
        results_getter=results_getter,
    )

    data_name = args.data_config.split("/")[-1].split(".")[0]
    pipeline_name = args.pipeline_config.split("/")[-1].split(".")[0]
    save_loc = f"{OUTPUT_DIR}/inference_results/{data_name}/{pipeline_name}"
    os.makedirs(save_loc, exist_ok=True)

    with open(f"{save_loc}/full_pipeline.json", "w") as save_file:
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
    args = parser.parse_args()

    main(args)
