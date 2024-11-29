import json
import os

from jsonargparse import CLI

from arc_spice.data.multieurlex_utils import load_multieurlex_for_translation
from arc_spice.eval.inference_utils import ResultsGetter, run_inference
from arc_spice.utils import open_yaml_path
from arc_spice.variational_pipelines.RTC_variational_pipeline import (
    RTCVariationalPipeline,
)

OUTPUT_DIR = "outputs"


def main(pipeline_config_pth: str, data_config_pth: str):
    """
    Run inference on a given pipeline with provided data config

    Args:
        pipeline_config_pth: path to pipeline config yaml file
        data_config_pth: path to data config yaml file
    """
    # initialise pipeline
    data_config = open_yaml_path(data_config_pth)
    pipeline_config = open_yaml_path(pipeline_config_pth)
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

    data_name = data_config_pth.split("/")[-1].split(".")[0]
    pipeline_name = pipeline_config_pth.split("/")[-1].split(".")[0]
    save_loc = f"{OUTPUT_DIR}/inference_results/{data_name}/{pipeline_name}"
    os.makedirs(save_loc, exist_ok=True)

    with open(f"{save_loc}/full_pipeline.json", "w") as save_file:
        json.dump(test_results, save_file)


if __name__ == "__main__":
    CLI(main)
