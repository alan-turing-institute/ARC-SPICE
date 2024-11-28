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

import json
import os

from jsonargparse import CLI

from arc_spice.data.multieurlex_utils import load_multieurlex_for_translation
from arc_spice.eval.inference_utils import ResultsGetter, run_inference
from arc_spice.utils import open_yaml_path
from arc_spice.variational_pipelines.RTC_single_component_pipeline import (
    ClassificationVariationalPipeline,
    RecognitionVariationalPipeline,
    TranslationVariationalPipeline,
)

OUTPUT_DIR = "outputs"


def main(pipeline_config_pth: str, data_config_pth: str, model_key: str):
    """
    Run inference on a given pipeline component with provided data config and model key.

    Args:
        pipeline_config_pth: path to pipeline config yaml file
        data_config_pth: path to data config yaml file
        model_key: name of model on which to run inference
    """
    # initialise pipeline
    data_config = open_yaml_path(data_config_pth)
    pipeline_config = open_yaml_path(pipeline_config_pth)
    data_sets, meta_data = load_multieurlex_for_translation(**data_config)
    test_loader = data_sets["test"]
    if model_key == "ocr":
        rtc_single_component_pipeline = RecognitionVariationalPipeline(
            model_pars=pipeline_config, data_pars=meta_data
        )
    elif model_key == "translator":
        rtc_single_component_pipeline = TranslationVariationalPipeline(
            model_pars=pipeline_config, data_pars=meta_data
        )
    elif model_key == "classifier":
        rtc_single_component_pipeline = ClassificationVariationalPipeline(
            model_pars=pipeline_config, data_pars=meta_data
        )
    else:
        error_msg = (
            "model_key should be: 'ocr', 'translator', or 'classifier'."
            f" Given: {model_key}"
        )
        raise ValueError(error_msg)

    results_getter = ResultsGetter(meta_data["n_classes"])

    test_results = run_inference(
        dataloader=test_loader,
        pipeline=rtc_single_component_pipeline,
        results_getter=results_getter,
    )

    data_name = data_config_pth.split("/")[-1].split(".")[0]
    pipeline_name = pipeline_config_pth.split("/")[-1].split(".")[0]
    save_loc = (
        f"{OUTPUT_DIR}/inference_results/{data_name}/{pipeline_name}/"
        f"single_component"
    )
    os.makedirs(save_loc, exist_ok=True)

    with open(f"{save_loc}/{model_key}.json", "w") as save_file:
        json.dump(test_results, save_file)


if __name__ == "__main__":
    CLI(main)
