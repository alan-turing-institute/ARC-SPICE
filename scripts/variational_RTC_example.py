"""
    An example use of the transcription, translation and summarisation pipeline.
"""

from random import randint

import torch
from torch.nn.functional import binary_cross_entropy

from arc_spice.data.multieurlex_dataloader import MultiHot, load_multieurlex
from arc_spice.eval.classification_error import hamming_accuracy
from arc_spice.eval.translation_error import get_comet_model
from arc_spice.variational_pipelines.RTC_variational_pipeline import (
    RTCVariationalPipeline,
)


def main(RTC_pars):
    # load data
    lang_pair = {"source": "fr", "target": "en"}
    [train, _, _], metadata_params = load_multieurlex(level=1, lang_pair=lang_pair)
    row_iterator = iter(train)
    multi_onehot = MultiHot(metadata_params["n_classes"])
    test_row = next(row_iterator)

    class_labels = multi_onehot(test_row["class_labels"])

    # load model
    RTC = RTCVariationalPipeline(RTC_pars, metadata_params)
    RTC.check_dropout()

    # run inference
    RTC.variational_inference(test_row["source_text"])

    # ### TRANSLATION ###
    print("\nTranslation:")
    source_text = test_row["target_text"]
    target_text = test_row["target_text"]
    clean_translation = RTC.clean_output["translation"]["full_output"]
    print(
        f"Semantic density: {RTC.var_output['translation']['weighted_semantic_density']}"
    )

    # load error model
    comet_model = get_comet_model()
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
    mean_scores = RTC.var_output["classification"]["mean_scores"]
    print(f"BCE: {binary_cross_entropy(mean_scores.float(), class_labels.float())}")
    preds = torch.round(mean_scores)
    hamming_acc = hamming_accuracy(preds=preds, class_labels=class_labels)
    print(f"hamming accuracy: {hamming_acc}")

    mean_entropy = torch.mean(RTC.var_output["classification"]["predicted_entropy"])
    mean_variances = torch.mean(RTC.var_output["classification"]["var_scores"])
    mean_MI = torch.mean(RTC.var_output["classification"]["mutual_information"])

    print("Predictive entropy: " f"{mean_entropy}")
    print("MI (model uncertainty): " f"{mean_MI}")
    print("Variance (model uncertainty): " f"{mean_variances}")


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
