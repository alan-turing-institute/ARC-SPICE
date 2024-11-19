"""
An example use of the transcription, translation and summarisation pipeline.
"""

import logging
from random import randint

import torch
from torch.nn.functional import binary_cross_entropy

from arc_spice.data.multieurlex_utils import MultiHot, load_multieurlex
from arc_spice.eval.classification_error import hamming_accuracy
from arc_spice.eval.translation_error import get_comet_model
from arc_spice.utils import seed_everything
from arc_spice.variational_pipelines.RTC_variational_pipeline import (
    RTCVariationalPipeline,
)


def load_test_row():
    lang_pair = {"source": "fr", "target": "en"}
    (train, _, _), metadata_params = load_multieurlex(
        data_dir="data", level=1, lang_pair=lang_pair
    )
    multi_onehot = MultiHot(metadata_params["n_classes"])
    test_row = get_test_row(train)
    class_labels = multi_onehot(test_row["labels"])
    return test_row, class_labels, metadata_params


def get_test_row(train_data):
    row_iterator = iter(train_data)
    for _ in range(randint(1, 25)):
        test_row = next(row_iterator)

    # debug row if needed
    return {
        "source_text": (
            "Le renard brun rapide a sauté par-dessus le chien paresseux."
            "Le renard a sauté par-dessus le chien paresseux."
        ),
        "target_text": (
            "The quick brown fox jumped over the lazy dog. The fox jumped"
            " over the lazy dog"
        ),
        "labels": [0, 1],
    }
    # Normal row
    return test_row


def print_results(clean_output, var_output, class_labels, test_row, comet_model):
    # ### TRANSLATION ###
    print("\nTranslation:")
    source_text = test_row["target_text"]
    target_text = test_row["target_text"]
    clean_translation = clean_output["translation"]["full_output"]
    print(f"Semantic density: {var_output['translation']['weighted_semantic_density']}")

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
    print(f"COMET: {comet_scores[0]}")

    # ### CLASSIFICATION ###
    print("\nClassification:")
    mean_scores = var_output["classification"]["mean_scores"]
    print(f"BCE: {binary_cross_entropy(mean_scores.float(), class_labels.float())}")
    preds = torch.round(mean_scores)
    hamming_acc = hamming_accuracy(preds=preds, class_labels=class_labels)
    print(f"hamming accuracy: {hamming_acc}")

    mean_entropy = torch.mean(var_output["classification"]["predicted_entropy"])
    mean_variances = torch.mean(var_output["classification"]["var_scores"])
    mean_MI = torch.mean(var_output["classification"]["mutual_information"])

    print("Predictive entropy: " f"{mean_entropy}")
    print("MI (model uncertainty): " f"{mean_MI}")
    print("Variance (model uncertainty): " f"{mean_variances}")


def main(rtc_pars):
    seed_everything(seed=42)

    logging.basicConfig(level=logging.INFO)

    test_row, class_labels, metadata_params = load_test_row()

    # initialise pipeline
    rtc_variational_pipeline = RTCVariationalPipeline(rtc_pars, metadata_params)

    # check dropout exists
    rtc_variational_pipeline.check_dropout()

    # perform variational inference
    clean_output, var_output = rtc_variational_pipeline.variational_inference(test_row)

    comet_model = get_comet_model()

    print_results(clean_output, var_output, class_labels, test_row, comet_model)


if __name__ == "__main__":
    rtc_pars = {
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
            "model": "claritylab/zero-shot-explicit-binary-bert",
        },
    }

    main(rtc_pars=rtc_pars)
