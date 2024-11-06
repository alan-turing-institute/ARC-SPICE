"""
    An example use of the transcription, translation and summarisation pipeline.
"""

from random import randint

import torch
from comet import download_model, load_from_checkpoint
from torch.nn.functional import binary_cross_entropy
from tqdm import tqdm

from arc_spice.data.multieurlex_dataloader import MultiOneHot, load_multieurlex
from arc_spice.eval.translation_error import get_bleu_score
from arc_spice.variational_pipelines.RTC_variational_pipeline import (
    RTCVariationalPipeline,
)

MAX_LEN = 256


def main(RTC_pars):
    lang_pair = {"source": "fr", "target": "en"}
    [train, _, _], metadata_params = load_multieurlex(level=1, lang_pair=lang_pair)
    row_iterator = iter(train)
    max_n_classes = 0
    for row in tqdm(row_iterator):
        if len(row["class_labels"]) > max_n_classes:
            max_n_classes = len(row["class_labels"])
    print(max_n_classes)
    multi_onehot = MultiOneHot(metadata_params["n_classes"])
    comet_model_pth = download_model(model="Unbabel/wmt22-comet-da")
    # or for example:
    # model_path = download_model("Unbabel/wmt22-comet-da")

    # Load the model checkpoint:
    comet_model = load_from_checkpoint(comet_model_pth)

    for _ in range(0, randint(1, 25)):
        test_row = next(row_iterator)

    class_labels = multi_onehot(test_row["class_labels"])

    RTC = RTCVariationalPipeline(RTC_pars, metadata_params)
    RTC.check_dropout()

    # RTC.variational_inference(test_row["source_text"])

    # print(RTC.var_output["translation"]["weighted_semantic_density"])
    # print(RTC.var_output["classification"])
    # mean_scores = RTC.var_output["classification"]["mean_scores"]
    # print(mean_scores)
    # print(binary_cross_entropy(mean_scores.float(), class_labels.float()))
    # preds = torch.round(mean_scores)
    # print(torch.mean((preds.float() == class_labels.float()).float()))

    # source_text = test_row["target_text"]
    # target_text = test_row["target_text"]
    # clean_translation = RTC.clean_output["translation"]["full_output"]

    # print(target_text, clean_translation)

    # dl_bleu = get_bleu_score(
    #     target_text,
    #     [clean_translation],
    # )

    # print(f"BLEU: {dl_bleu}")

    # comet_inp = [
    #     {
    #         "src": source_text,
    #         "mt": clean_translation,
    #         "ref": target_text,
    #     }
    # ]

    # comet_output = comet_model.predict(comet_inp, batch_size=8, accelerator="cpu")
    # comet_scores = comet_output["scores"]

    # print(f"COMET: {comet_scores[0]}")


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
