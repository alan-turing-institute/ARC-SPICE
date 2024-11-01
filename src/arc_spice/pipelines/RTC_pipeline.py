"""
    Class for the transcription, translation and summarisation pipeline.
"""

import torch
from transformers import pipeline

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class RTCPipeline:
    """
    Class for the transcription, translation, summarisation pipeline.

    pars:
        - {'top_level_task': {'specific_task': str, 'model_name': str}}
    """

    def __init__(self, model_pars, data_pars) -> None:

        self.pars = model_pars
        self.OCR = pipeline(
            model_pars["OCR"]["specific_task"],
            model_pars["OCR"]["model"],
            device=device,
        )
        self.translator = pipeline(
            model_pars["translator"]["specific_task"],
            model_pars["translator"]["model"],
            max_length=512,
            device=device,
        )
        self.classifier = pipeline(
            model_pars["classifier"]["specific_task"],
            model_pars["classifier"]["model"],
            multi_label=True,
            device=device,
        )
        self.candidate_labels = [
            class_names_dict["en"]
            for class_names_dict in data_pars["class_descriptors"]
        ]
        self.max_translation_len = 128

    @staticmethod
    def split_inputs(text, split_key):
        split_rows = text.split(split_key)
        recovered_splits = [split + split_key for split in split_rows]
        return recovered_splits

    @staticmethod
    def stack_inputs(split_inputs, max_len):
        all_stacked_outputs = []
        stacked_output = ""
        for split in split_inputs:
            if len(stacked_output + split) < max_len:
                stacked_output += split
            else:
                all_stacked_outputs.append(stacked_output)
                stacked_output = split
        all_stacked_outputs.append(stacked_output)
        return all_stacked_outputs

    def translate(self, text):
        if len(text) >= self.max_translation_len:
            translation = ""
            text_splits = self.split_inputs(text, ".")

            for split in text_splits:
                translator_output = self.translator(split)
                translation += translator_output[0]["translation_text"]

            return translation

        return self.translator(text)[0]["translation_text"]

    def classify(self, text):
        return self.classifier(text, self.candidate_labels)

    def run_pipeline(self, x):
        self.results = {}
        """Run the pipeline on an input x"""
        recognition = self.OCR(x)
        self.results["OCR"] = recognition["text"]
        self.results["translation"] = self.translate(recognition["text"])
        self.results["classification"] = self.classify(self.results["translation"])
