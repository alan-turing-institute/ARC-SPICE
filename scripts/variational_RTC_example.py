"""
    An example use of the transcription, translation and summarisation pipeline.
"""

from random import randint
import random
import numpy as np
import os

import torch
from comet import download_model, load_from_checkpoint
from torch.nn.functional import binary_cross_entropy
from tqdm import tqdm

from arc_spice.data.multieurlex_dataloader import MultiHot, load_multieurlex
from arc_spice.eval.classification_error import (
    aggregate_score,
    hamming_accuracy,
)
from arc_spice.eval.translation_error import get_bleu_score
from arc_spice.variational_pipelines.RTC_variational_pipeline import (
    RTCVariationalPipeline,
)
import logging

MAX_LEN = 256


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_test_row():
    lang_pair = {"source": "fr", "target": "en"}
    (train, _, _), metadata_params = load_multieurlex(
        level=1, lang_pair=lang_pair
    )
    # TODOO refactor
    multi_onehot = MultiHot(metadata_params["n_classes"])
    test_row = get_test_row(train)
    class_labels = multi_onehot(test_row["class_labels"])
    return test_row, class_labels, metadata_params


def get_test_row(train_data):
    row_iterator = iter(train_data)
    for _ in range(0, randint(1, 25)):
        test_row = next(row_iterator)
    return test_row


# debug row if needed
# def get_test_row(train_data):
#     return {
#         "source_text": "Le renard brun rapide a sauté par-dessus le chien paresseux. Le renard a sauté par-dessus le chien paresseux.",
#         "target_text": "The quick brown fox jumped over the lazy dog. The fox jumped over the lazy dog",
#         "class_labels": [0, 1],
#     }


def load_comet():
    comet_model_pth = download_model(model="Unbabel/wmt22-comet-da")
    # Load the model checkpoint:
    return load_from_checkpoint(comet_model_pth)


def print_results(
    rtc_variational_pipeline, class_labels, test_row, comet_model
):
    print("\nClassification:")
    mean_scores = rtc_variational_pipeline.var_output["classification"][
        "mean_scores"
    ]
    print(
        f"BCE: {binary_cross_entropy(mean_scores.float(), class_labels.float())}"
    )
    preds = torch.round(mean_scores)
    hamming_acc = hamming_accuracy(preds=preds, class_labels=class_labels)
    print(f"hamming accuracy: {hamming_acc}")
    confidence_score = aggregate_score(probs=mean_scores)
    print(f"confidence score: {confidence_score}")

    print("\nTranslation:")
    source_text = test_row["target_text"]
    target_text = test_row["target_text"]
    clean_translation = rtc_variational_pipeline.clean_output["translation"][
        "full_output"
    ]
    print(
        f"Semantic density: {rtc_variational_pipeline.var_output['translation']['weighted_semantic_density']}"
    )
    comet_inp = [
        {
            "src": source_text,
            "mt": clean_translation,
            "ref": target_text,
        }
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    comet_output = comet_model.predict(
        comet_inp, batch_size=8, accelerator=device
    )
    comet_scores = comet_output["scores"]
    print(f"COMET: {comet_scores[0]}")


def main(RTC_pars):

    seed_everything(seed=42)

    logging.basicConfig(level=logging.INFO)

    test_row, class_labels, metadata_params = load_test_row()

    # initialise pipeline
    rtc_variational_pipeline = RTCVariationalPipeline(
        RTC_pars, metadata_params
    )

    # check dropout exists
    rtc_variational_pipeline.check_dropout()

    # perform variational inference
    rtc_variational_pipeline.variational_inference(test_row["source_text"])

    comet_model = load_comet()

    print_results(
        rtc_variational_pipeline, class_labels, test_row, comet_model
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
            "model": "claritylab/zero-shot-explicit-binary-bert",
        },
    }
    main(RTC_pars=RTC_pars)
