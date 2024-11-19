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

import torch
from comet.models import CometModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from arc_spice.data.multieurlex_utils import MultiHot, load_multieurlex
from arc_spice.eval.classification_error import hamming_accuracy
from arc_spice.eval.translation_error import get_comet_model
from arc_spice.utils import open_yaml_path
from arc_spice.variational_pipelines.RTC_variational_pipeline import (
    RTCVariationalPipeline,
)


class ResultsGetter:
    def __init__(self, n_classes):
        self.multi_hot = MultiHot(n_classes)

    def get_results(
        self,
        clean_output: dict[str, dict],
        var_output: dict[str, dict],
        test_row: dict[str, list[int] | str],
        comet_model: CometModel,
        results_dict: dict[str, dict[str, list[float]]],
    ):
        # ### RECOGNITION ###
        # TODO: add this into results_getter

        # ### TRANSLATION ###
        source_text = test_row["target_text"]
        target_text = test_row["target_text"]
        clean_translation = clean_output["translation"]["full_output"]

        # load error model
        comet_inp = [
            {
                "src": source_text,
                "mt": clean_translation,
                "ref": target_text,
            }
        ]
        # comet doesn't work on MPS
        comet_device = "cuda" if torch.cuda.is_available() else "cpu"
        comet_output = comet_model.predict(
            comet_inp, batch_size=8, accelerator=comet_device
        )
        comet_scores = comet_output["scores"]
        results_dict["translation"]["comet_score"].append(comet_scores[0])
        results_dict["translation"]["weighted_semantic_density"].append(
            var_output["translation"]["weighted_semantic_density"]
        )

        # ### CLASSIFICATION ###
        mean_scores = var_output["classification"]["mean_scores"]
        preds = torch.round(mean_scores)
        hamming_acc = hamming_accuracy(
            preds=preds, class_labels=self.multi_hot(test_row["labels"])
        )
        results_dict["classification"]["hamming_accuracy"].append(hamming_acc)
        results_dict["classification"]["mean_predicted_entropy"].append(
            torch.mean(var_output["classification"]["predicted_entropy"]).item()
        )

        return results_dict


def run_inference(
    dataloader: DataLoader,
    pipeline: RTCVariationalPipeline,
    results_getter: ResultsGetter,
    comet_model: CometModel,
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

    return results_dict


def main(args):
    # initialise pipeline
    data_config = open_yaml_path(args.data_config)
    pipeline_config = open_yaml_path(args.pipeline_config)
    (_, _, val_loader), meta_data = load_multieurlex(**data_config)
    rtc_variational_pipeline = RTCVariationalPipeline(
        model_pars=pipeline_config, data_pars=meta_data
    )
    comet_model = get_comet_model()
    results_getter = ResultsGetter(meta_data["n_classes"])

    val_results = run_inference(
        dataloader=val_loader,
        pipeline=rtc_variational_pipeline,
        results_getter=results_getter,
        comet_model=comet_model,
    )

    print(val_results)


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
