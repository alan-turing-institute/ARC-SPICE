from collections import namedtuple
from collections.abc import Callable
from typing import Any

import torch
from sklearn.metrics import hamming_loss, zero_one_loss
from torch.utils.data import DataLoader
from tqdm import tqdm

from arc_spice.data.multieurlex_utils import MultiHot
from arc_spice.eval.ocr_error import ocr_error
from arc_spice.eval.translation_error import get_comet_model
from arc_spice.variational_pipelines.RTC_single_component_pipeline import (
    RTCSingleComponentPipeline,
)
from arc_spice.variational_pipelines.RTC_variational_pipeline import (
    RTCVariationalPipeline,
)

RecognitionResults = namedtuple("RecognitionResults", ["confidence", "accuracy"])
ClassificationResults = namedtuple(
    "ClassificationResults",
    [
        "mean_scores",
        "hamming_accuracy",
        "zero_one_accuracy",
        "mean_predicted_entropy",
    ],
)
TranslationResults = namedtuple(
    "TranslationResults",
    [
        "full_output",
        "comet_score",
        "weighted_semantic_density",
    ],
)


class ResultsGetter:
    def __init__(self, n_classes: int):
        self.results_func_map: dict[
            str, Callable[..., ClassificationResults | TranslationResults]
        ] = {
            "recognition": self.recognition_results,
            "translation": self.translation_results,
            "classification": self.classification_results,
        }
        self.comet_model = get_comet_model()
        self.multihot = MultiHot(n_classes)

    def get_results(
        self,
        clean_output: dict[str, dict],
        var_output: dict[str, dict],
        test_row: dict[str, list[int]],
    ):
        results_dict = {}
        for step_name in clean_output:
            results_dict[step_name] = self.results_func_map[step_name](
                test_row=test_row,
                clean_output=clean_output,
                var_output=var_output,
            )._asdict()
        return results_dict

    def recognition_results(
        self,
        clean_output: dict[str, str | list[dict[str, str | torch.Tensor]]],
        var_output: dict[str, dict],
        **kwargs,
    ):
        # ### RECOGNITION ###
        charerror = ocr_error(clean_output)
        confidence = var_output["recognition"]["mean_entropy"]
        return RecognitionResults(confidence=confidence, accuracy=charerror)

    def translation_results(
        self,
        test_row: dict[str, Any],
        clean_output: dict[str, dict],
        var_output: dict[str, dict],
    ):
        # ### TRANSLATION ###
        source_text = test_row["target_text"]
        target_text = test_row["target_text"]
        clean_translation = clean_output["translation"]["full_output"]

        # define error model inputs
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

        return TranslationResults(
            comet_score=comet_output["scores"][0],
            full_output=clean_translation,
            weighted_semantic_density=var_output["translation"][
                "weighted_semantic_density"
            ],
        )

    def classification_results(
        self,
        test_row: dict[str, Any],
        var_output: dict[str, dict],
        **kwargs,
    ):
        # ### CLASSIFICATION ###
        mean_scores: torch.Tensor = var_output["classification"]["mean_scores"]
        preds = torch.round(mean_scores).tolist()
        labels = self.multihot(test_row["labels"])
        hamming_acc = hamming_loss(y_pred=preds, y_true=labels)
        zero_one_acc = zero_one_loss(y_pred=preds, y_true=labels)

        return ClassificationResults(
            mean_scores=mean_scores.detach().tolist(),
            hamming_accuracy=hamming_acc,
            zero_one_accuracy=zero_one_acc,
            mean_predicted_entropy=torch.mean(
                var_output["classification"]["predicted_entropy"]
            ).item(),
        )


def run_inference(
    dataloader: DataLoader,
    pipeline: RTCVariationalPipeline | RTCSingleComponentPipeline,
    results_getter: ResultsGetter,
):
    results = []
    for _, inp in enumerate(tqdm(dataloader)):
        clean_out, var_out = pipeline.variational_inference(inp)
        row_results_dict = results_getter.get_results(
            clean_output=clean_out,
            var_output=var_out,
            test_row=inp,
        )
        results.append({inp["celex_id"]: row_results_dict})
    return results
