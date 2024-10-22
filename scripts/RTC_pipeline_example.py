"""
    An example use of the transcription, translation and summarisation pipeline.
"""

import torch
from torch.nn.functional import softmax

from arc_spice.data.multieurlex_dataloader import load_multieurlex
from arc_spice.pipelines.RTC_pipeline import RTCPipeline


def main(RTC_pars):
    lang_pair = {"source": "fr", "target": "en"}
    [train, _, _], metadata_params = load_multieurlex(level=1, lang_pair=lang_pair)
    RTC = RTCPipeline(RTC_pars, metadata_params)
    test_row = next(iter(train))
    print(test_row["source_text"])
    print(test_row["target_text"])
    translator_output = RTC.translator(test_row["source_text"])
    translation = translator_output[0]["translation_text"]
    classifier_output = RTC.classifier(translation, RTC.candidate_labels)
    classification = classifier_output["scores"]

    print(translation)
    print(classification)
    # print(translation[0]["translation_text"])
    # print(test_row["target_text"])


if __name__ == "__main__":
    RTC_pars = {
        "OCR": {
            "specific_task": "image-to-text",
            "model": "microsoft/trocr-base-handwritten",
        },
        "translator": {
            "specific_task": "translation_fr_to_en",
            "model": "facebook/mbart-large-50-many-to-many-mmt",
        },
        "classifier": {
            "specific_task": "zero-shot-classification",
            "model": "facebook/bart-large-mnli",
        },
    }
    main(RTC_pars=RTC_pars)
