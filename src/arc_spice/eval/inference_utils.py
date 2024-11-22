from collections.abc import Callable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from arc_spice.data.multieurlex_utils import MultiHot
from arc_spice.eval.classification_error import hamming_accuracy
from arc_spice.eval.translation_error import get_comet_model
from arc_spice.variational_pipelines.RTC_single_component_pipeline import (
    RTCSingleComponentPipeline,
)
from arc_spice.variational_pipelines.RTC_variational_pipeline import (
    RTCVariationalPipeline,
)


class ResultsGetter:
    def __init__(self, n_classes):
        self.multi_hot = MultiHot(n_classes)
        self.func_map: dict[str, Callable] = {
            "recognition": self.recognition_results,
            "translation": self.translation_results,
            "classification": self.classification_results,
        }
        self.comet_model = get_comet_model()

    def get_results(
        self,
        clean_output: dict[str, dict],
        var_output: dict[str, dict],
        test_row: dict[str, list[int]],
        results_dict: dict[str, dict[str, list[float]]],
    ):
        for step_name in clean_output:
            results_dict = self.func_map[step_name](
                test_row, clean_output, var_output, results_dict
            )

        return results_dict

    def single_step_results(
        self,
        clean_output: dict[str, dict],
        var_output: dict[str, dict],
        test_row: dict[str, list[int]],
        results_dict: dict[str, dict[str, list[float]]],
        step_name: str,
    ):
        return self.func_map[step_name](
            test_row, clean_output, var_output, results_dict
        )

    def recognition_results(self, test_row, clean_output, var_output, results_dict):
        assert test_row is not None
        assert clean_output is not None
        assert var_output is not None
        # ### RECOGNITION ###
        # TODO: add this into results_getter
        return results_dict

    def translation_results(self, test_row, clean_output, var_output, results_dict):
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
        comet_output = self.comet_model.predict(
            comet_inp, batch_size=8, accelerator=comet_device
        )
        comet_scores = comet_output["scores"]
        results_dict["translation"]["comet_score"].append(comet_scores[0])
        results_dict["translation"]["weighted_semantic_density"].append(
            var_output["translation"]["weighted_semantic_density"]
        )
        return results_dict

    def classification_results(self, test_row, _, var_output, results_dict):
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
    pipeline: RTCVariationalPipeline | RTCSingleComponentPipeline,
    results_getter: ResultsGetter,
):
    results_dict = {
        # Placeholder
        "ocr": {"confidence": [], "accuracy": []},
        "translation": {"weighted_semantic_density": [], "comet_score": []},
        "classification": {"mean_predicted_entropy": [], "hamming_accuracy": []},
    }
    if isinstance(pipeline, RTCSingleComponentPipeline):
        results_dict = {pipeline.step_name: results_dict[pipeline.step_name]}

    for _, inp in enumerate(tqdm(dataloader)):
        clean_out, var_out = pipeline.variational_inference(inp)
        results_dict = results_getter.get_results(
            clean_output=clean_out,
            var_output=var_out,
            test_row=inp,
            results_dict=results_dict,
        )

    return results_dict
