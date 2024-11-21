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

from comet.models import CometModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from arc_spice.data.multieurlex_utils import load_multieurlex
from arc_spice.eval.inference_utils import ResultsGetter
from arc_spice.eval.translation_error import get_comet_model
from arc_spice.utils import open_yaml_path
from arc_spice.variational_pipelines.RTC_single_component_pipeline import (
    RTCSingleComponentPipeline,
)


def run_inference(
    dataloader: DataLoader,
    pipeline: RTCSingleComponentPipeline,
    results_getter: ResultsGetter,
    comet_model: CometModel | None,
):
    results_dict = {
        # Placeholder
        "ocr": {"confidence": [], "accuracy": []},
        "translation": {"weighted_semantic_density": [], "comet_score": []},
        "classification": {"mean_predicted_entropy": [], "hamming_accuracy": []},
    }
    for _, inp in enumerate(tqdm(dataloader)):
        clean_out, var_out = pipeline.variational_inference(inp)
        results_dict = results_getter.get_results(
            clean_output=clean_out,
            var_output=var_out,
            test_row=inp,
            comet_model=comet_model,
            results_dict=results_dict,
        )
        break

    return results_dict


def main(args):
    # initialise pipeline
    data_config = open_yaml_path(args.data_config)
    pipeline_config = open_yaml_path(args.pipeline_config)
    (_, test_loader, _), meta_data = load_multieurlex(**data_config)
    rtc_single_component_pipeline = RTCSingleComponentPipeline(
        model_pars=pipeline_config, data_pars=meta_data, model_key=args.model_key
    )
    comet_model = get_comet_model()
    results_getter = ResultsGetter(meta_data["n_classes"])

    test_results = run_inference(
        dataloader=test_loader,
        pipeline=rtc_single_component_pipeline,
        results_getter=results_getter,
        comet_model=comet_model,
    )

    print(test_results)


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
