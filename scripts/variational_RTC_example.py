"""
    An example use of the transcription, translation and summarisation pipeline.
"""

from arc_spice.data.multieurlex_dataloader import load_multieurlex
from arc_spice.variational_pipelines.RTC_variational_pipeline import (
    RTCVariationalPipeline,
)

# special_split = RTCPipeline.split_inputs
# stack = RTCPipeline.stack_inputs

MAX_LEN = 256


def main(RTC_pars):
    lang_pair = {"source": "fr", "target": "en"}
    [train, _, _], metadata_params = load_multieurlex(level=1, lang_pair=lang_pair)

    row_iterator = iter(train)
    test_row = next(row_iterator)

    # split_source = special_split(test_row["source_text"], ".")
    # split_target = special_split(test_row["target_text"], ".")

    # for index, (source, target) in enumerate(zip(split_source, split_target)):
    #     print(f"input {index}")
    #     print(f"source:\n{source}\n")
    #     print(f"target:\n{target}\n")

    # stacked_source = stack(split_source, MAX_LEN)
    # stacked_target = stack(split_target, MAX_LEN)

    # for index, (source, target) in enumerate(zip(stacked_source, stacked_target)):
    #     print(f"input {index}")
    #     print(f"source:\n{source}\n")
    #     print(f"target:\n{target}\n")

    # print(test_row["source_text"])
    # print(test_row["target_text"])

    RTC = RTCVariationalPipeline(RTC_pars, metadata_params)
    # translation = RTC.translate(test_row["source_text"])
    # classifier_output = RTC.classify(translation)
    # classification = classifier_output["scores"]

    # print(translation)
    # print(classification)

    # RTC.clean_inference(test_row["source_text"])
    # # print(RTC.clean_output["translation"]["outputs"])
    # # print(RTC.clean_output["translation"]["semantic_embedding"])
    # print(RTC.clean_output["classification"])

    RTC.variational_inference(test_row["source_text"])

    print(RTC.var_output["variational"]["translation"]["full_output"])


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
            "model": "facebook/bart-large-mnli",
        },
    }
    main(RTC_pars=RTC_pars)
