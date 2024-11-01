"""
    An example use of the transcription, translation and summarisation pipeline.
"""

import torch
from torch.nn.functional import binary_cross_entropy

from arc_spice.data.multieurlex_dataloader import MultiOneHot, load_multieurlex
from arc_spice.eval.translation_error import get_bleu_score
from arc_spice.variational_pipelines.RTC_variational_pipeline import (
    RTCVariationalPipeline,
)

# special_split = RTCPipeline.split_inputs
# stack = RTCPipeline.stack_inputs

MAX_LEN = 256


def main(RTC_pars):
    lang_pair = {"source": "fr", "target": "en"}
    [train, _, _], metadata_params = load_multieurlex(level=1, lang_pair=lang_pair)
    multi_onehot = MultiOneHot(metadata_params["n_classes"])

    row_iterator = iter(train)
    test_row = next(row_iterator)

    class_labels = multi_onehot(test_row["class_labels"])
    print(class_labels)

    RTC = RTCVariationalPipeline(RTC_pars, metadata_params)
    RTC.check_dropout()
    RTC.variational_inference(test_row["source_text"])

    print(RTC.var_output["translation"]["weighted_semantic_density"])
    print(RTC.var_output["classification"])
    mean_scores = RTC.var_output["classification"]["mean_scores"]
    print(mean_scores)
    print(binary_cross_entropy(mean_scores.float(), class_labels.float()))
    preds = torch.round(mean_scores)
    print(torch.mean((preds.float() == class_labels.float()).float()))

    print(
        get_bleu_score(
            test_row["target_text"], [RTC.clean_output["translation"]["full_output"]]
        )
    )


if __name__ == "__main__":
    RTC_pars = {
        "OCR": {
            "specific_task": "image-to-text",
            "model": "microsoft/trocr-base-handwritten",
        },
        "translator": {
            "specific_task": "translation_fr_to_en",
            "model": "ybanas/autotrain-fr-en-translate-51410121895",
        },
        "classifier": {
            "specific_task": "zero-shot-classification",
            # "model": "cross-encoder/nli-MiniLM2-L6-H768",
            # "model": "cointegrated/rubert-base-cased-nli-threeway",
            "model": "claritylab/zero-shot-explicit-binary-bert",
        },
    }
    main(RTC_pars=RTC_pars)
