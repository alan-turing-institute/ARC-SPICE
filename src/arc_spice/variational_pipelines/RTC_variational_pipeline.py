from typing import Any

import torch
from transformers import pipeline

from arc_spice.variational_pipelines.utils import (
    CustomOCRPipeline,
    CustomTranslationPipeline,
    RTCVariationalPipelineBase,
    dropout_off,
    dropout_on,
    set_classifier,
    set_dropout,
)


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# OCR, Translationslation, Topic Classification
class RTCVariationalPipeline(RTCVariationalPipelineBase):
    """
    variational version of the RTC pipeline
    """

    def __init__(
        self,
        model_pars: dict[str, dict[str, str]],
        data_pars: dict[str, Any],
        n_variational_runs=5,
        translation_batch_size=16,
        ocr_batch_size=64,
    ) -> None:
        # are we doing zero-shot-classification?
        if model_pars["classifier"]["specific_task"] == "zero-shot-classification":
            self.zero_shot = True
        else:
            self.zero_shot = False
        super().__init__(self.zero_shot, n_variational_runs, translation_batch_size)
        # defining the pipeline objects
        self.ocr = pipeline(
            model=model_pars["OCR"]["model"],
            device=self.device,
            pipeline_class=CustomOCRPipeline,
            max_new_tokens=20,
            batch_size=ocr_batch_size,
        )
        self.translator = pipeline(
            task=model_pars["translator"]["specific_task"],
            model=model_pars["translator"]["model"],
            max_length=512,
            pipeline_class=CustomTranslationPipeline,
            device=self.device,
        )
        self.classifier = set_classifier(model_pars["classifier"], self.device)
        # topic meta_data for the classifier
        self.dataset_meta_data = data_pars
        self._init_semantic_density()
        self._init_pipeline_map()

    def clean_inference(self, x: torch.Tensor) -> dict[str, dict]:
        """Run the pipeline on an input x"""
        # define output dictionary
        clean_output: dict[str, Any] = {
            "recognition": {},
            "translation": {},
            "classification": {},
        }

        # run the functions
        # UNTIL THE OCR DATA IS AVAILABLE
        clean_output["recognition"] = self.recognise(x)

        clean_output["translation"] = self.translate(
            clean_output["recognition"]["outputs"]
        )
        # we now need to pass the input correct to the correct forward method
        if self.zero_shot:
            clean_output["classification"] = self.classify_topic_zero_shot(
                clean_output["translation"]["outputs"][0]
            )
        else:
            clean_output["classification"] = self.classify_topic(
                clean_output["translation"]["outputs"][0]
            )
        return clean_output

    def variational_inference(self, x: torch.Tensor) -> tuple[dict, dict]:
        """
        runs the variational inference with the pipeline
        """
        # ...first run clean inference
        clean_output = self.clean_inference(x)
        # define output dictionary
        var_output: dict[str, Any] = {
            "recognition": [None] * self.n_variational_runs,
            "translation": [None] * self.n_variational_runs,
            "classification": [None] * self.n_variational_runs,
        }
        # define the input map for brevity in forward pass
        input_map = {
            "recognition": x,
            "translation": clean_output["recognition"]["outputs"],
            "classification": clean_output["translation"]["full_output"],
        }

        # for each model in pipeline
        for model_key, pl in self.pipeline_map.items():
            # turn on dropout for this model
            set_dropout(model=pl.model, dropout_flag=True)  # type: ignore[union-attr,attr-defined]
            torch.nn.functional.dropout = dropout_on
            # do n runs of the inference
            for run_idx in range(self.n_variational_runs):
                var_output[model_key][run_idx] = self.func_map[model_key](
                    input_map[model_key]
                )
            # turn off dropout for this model
            set_dropout(model=pl.model, dropout_flag=False)  # type: ignore[union-attr,attr-defined]
            torch.nn.functional.dropout = dropout_off

        # run metric helper functions
        var_output = self.stack_variational_outputs(var_output)
        var_output = self.translation_semantic_density(
            clean_output=clean_output, var_output=var_output
        )
        var_output = self.get_classification_confidence(var_output)

        return clean_output, var_output

    # on standard call return the clean output
    def __call__(self, x):
        return self.clean_inference(x)
