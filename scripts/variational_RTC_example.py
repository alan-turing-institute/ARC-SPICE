"""
    An example use of the transcription, translation and summarisation pipeline.
"""

from random import randint

import torch
from comet import download_model, load_from_checkpoint
from torch.nn.functional import binary_cross_entropy
from tqdm import tqdm

from arc_spice.data.multieurlex_dataloader import MultiHot, load_multieurlex
from arc_spice.eval.classification_error import aggregate_score, hamming_accuracy
from arc_spice.eval.translation_error import get_bleu_score
from arc_spice.variational_pipelines.RTC_variational_pipeline import (
    RTCVariationalPipeline,
)

MAX_LEN = 256


def main(RTC_pars):
    lang_pair = {"source": "fr", "target": "en"}
    [train, _, _], metadata_params = load_multieurlex(level=1, lang_pair=lang_pair)
    row_iterator = iter(train)
    multi_onehot = MultiHot(metadata_params["n_classes"])
    comet_model_pth = download_model(model="Unbabel/wmt22-comet-da")

    # Load the model checkpoint:
    comet_model = load_from_checkpoint(comet_model_pth)

    for _ in range(0, randint(1, 25)):
        test_row = next(row_iterator)

    class_labels = multi_onehot(test_row["class_labels"])

    RTC = RTCVariationalPipeline(RTC_pars, metadata_params)
    RTC.check_dropout()
    RTC.variational_inference(test_row["source_text"])

    with open("temp/test_output.txt", "w") as text_file:
        text_file.write("Purchase Amount: %s" % test_row["source_text"])

    exit()

    print("\nClassification:")
    mean_scores = RTC.var_output["classification"]["mean_scores"]
    print(f"BCE: {binary_cross_entropy(mean_scores.float(), class_labels.float())}")
    preds = torch.round(mean_scores)
    hamming_acc = hamming_accuracy(preds=preds, class_labels=class_labels)
    print(f"hamming accuracy: {hamming_acc}")
    confidence_score = aggregate_score(probs=mean_scores)
    print(f"confidence score: {confidence_score}")

    print("\nTranslation:")
    source_text = test_row["target_text"]
    target_text = test_row["target_text"]
    clean_translation = RTC.clean_output["translation"]["full_output"]
    print(
        f"Semantic density: {RTC.var_output['translation']['weighted_semantic_density']}"
    )
    comet_inp = [
        {
            "src": source_text,
            "mt": clean_translation,
            "ref": target_text,
        }
    ]
    comet_output = comet_model.predict(comet_inp, batch_size=8, accelerator="cpu")
    comet_scores = comet_output["scores"]
    print(f"COMET: {comet_scores[0]}")


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
