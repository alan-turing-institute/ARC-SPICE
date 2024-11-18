"""
Steps:
    - Load data
    - Load pipeline
    - Run inference on all test data
        - Save outputs of each model (on clean and pipeline output)
        - Calculate error of each model (on clean and pipeline output)

    - Save results
        - File structure:
            - output/check_callibration/pipeline_name/run_[X]/[OUTPUT FILES HERE]
"""

import argparse

from arc_spice.data.multieurlex_utils import load_multieurlex
from arc_spice.utils import open_yaml_path
from arc_spice.variational_pipelines.RTC_variational_pipeline import (
    RTCVariationalPipeline,
)


def main(args):
    # initialise pipeline
    data_config = open_yaml_path(args.data_config)
    pipeline_config = open_yaml_path(args.pipeline_config)
    (train_set, test_set, val_set), meta_data = load_multieurlex(**data_config)
    rtc_variational_pipeline = RTCVariationalPipeline(
        model_pars=pipeline_config, data_pars=meta_data
    )
    print(rtc_variational_pipeline)


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
