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

        # ################################################## #
        # This is just a hotfix until we can get mps working #
        device = "cpu"
        # ################################################## #

        self.pars = model_pars
        self.OCR = pipeline(
            model_pars["OCR"]["specific_task"],
            model_pars["OCR"]["model"],
            device=device,
        )
        self.translator = pipeline(
            model_pars["translator"]["specific_task"],
            model_pars["translator"]["model"],
            max_length=1000,
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

    def run_pipeline(self, x):
        self.results = {}
        """Run the pipeline on an input x"""
        recognition = self.OCR(x)
        self.results["OCR"] = recognition["text"]

        translation = self.translator(recognition["text"])
        self.results["translation"] = translation[0]["translation_text"]

        classification = self.classifier(
            translation[0]["translation_text"], self.candidate_labels
        )
        self.results["classification"] = classification[0]["output"]
