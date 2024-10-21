"""
    An example use of the transcription, translation and summarisation pipeline.
"""

import json

from datasets import Audio, load_dataset

from arc_spice.data.multieurlex_dataloader import load_multieurlex
from arc_spice.pipelines.RTC_pipeline import RTCPipeline


def main(TTS_params):
    lang_pair = {"source": "fr", "target": "en"}
    [train, test, val], metadata_params = load_multieurlex(level=1, lang_pair=lang_pair)
    RTC = RTCPipeline(TTS_params, metadata_params)
    test_row = next(iter(train))

    translation = RTC.translator(test_row["source_text"])
    print(translation[0]["translation_text"])
    print(test_row["target_text"])


if __name__ == "__main__":
    TTS_pars = {
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
    main(TTS_params=TTS_pars)
