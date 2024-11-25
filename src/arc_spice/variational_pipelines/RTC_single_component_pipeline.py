from collections.abc import Callable
from typing import Any

import torch
from transformers import pipeline

from arc_spice.variational_pipelines.dropout_utils import (
    dropout_off,
    dropout_on,
    set_dropout,
)
from arc_spice.variational_pipelines.RTC_variational_pipeline import (
    CustomTranslationPipeline,
    RTCVariationalPipeline,
)


class RTCSingleComponentPipeline(RTCVariationalPipeline):
    """
    Single component version of the variational pipeline, which inherits methods from
    the main `RTCVariationalPipeline` class, without initialising models by overwriting
    the `.__init__` method, the `.clean_inference` and `.variational_inference`
    methods are also overwitten accordingly.

    Args:
        RTCVariationalPipeline: Inherits from the variational pipeline
    """

    def __init__(
        self,
        model_pars: dict[str, dict[str, str]],
        model_key: str,
        data_pars: dict[str, Any],
        n_variational_runs: int = 5,
        translation_batch_size: int = 8,
    ) -> None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        # define objects that are needed and nothing else
        if model_key == "OCR":
            self.step_name = "recognition"
            self.input_key = "ocr_data"
            self.ocr = pipeline(
                task=model_pars["OCR"]["specific_task"],
                model=model_pars["OCR"]["model"],
                device=device,
            )
            self.model = self.ocr.model

        elif model_key == "translator":
            self.step_name = "translation"
            self.input_key = "source_text"
            self.translation_batch_size = translation_batch_size
            self.translator = pipeline(
                task=model_pars["translator"]["specific_task"],
                model=model_pars["translator"]["model"],
                max_length=512,
                pipeline_class=CustomTranslationPipeline,
                device=device,
            )
            self.model = self.translator.model
            # need to initialise the NLI models in this case
            self._init_semantic_density()

        elif model_key == "classifier":
            self.step_name = "classification"
            self.input_key = "target_text"
            self.classifier = pipeline(
                task=model_pars["classifier"]["specific_task"],
                model=model_pars["classifier"]["model"],
                multi_label=True,
                device=device,
            )
            self.model = self.classifier.model
            # topic description labels for the classifier
            self.topic_labels = [
                class_names_dict["en"]
                for class_names_dict in data_pars["class_descriptors"]
            ]
        # if an incorrect model is not selected
        else:
            error_msg = "Please specify a valid pipeline component"
            raise ValueError(error_msg)

        # naive outputs can remain the same, though only the appropriate outputs will
        # be outputted
        self.naive_outputs = {
            "recognition": [
                "outputs",
            ],
            "translation": [
                "full_output",
                "outputs",
                "probs",
            ],
            "classification": [
                "scores",
            ],
        }

        # maps to ensure the correct foward method is used in inference.
        self.foward_func_map: dict[str, Callable] = {
            "recognition": self.recognise,
            "translation": self.translate,
            "classification": self.classify_topic,
        }
        self.confidence_func_map: dict[str, Callable] = {
            "recognition": self.recognise,
            "translation": self.translation_semantic_density,
            "classification": self.get_classification_confidence,
        }

        self.n_variational_runs = n_variational_runs

    # clean inference
    def clean_inference(self, x):
        inp = x[self.input_key]
        clean_output: dict[str, Any] = {
            self.step_name: {},
        }
        clean_output[self.step_name] = self.foward_func_map[self.step_name](inp)
        return clean_output

    def variational_inference(self, x):
        clean_output = self.clean_inference(x)
        inp = x[self.input_key]
        var_output: dict[str, Any] = {
            self.step_name: {},
        }
        # turn on dropout for this model
        set_dropout(model=self.model, dropout_flag=True)
        torch.nn.functional.dropout = dropout_on
        # do n runs of the inference
        for run_idx in range(self.n_variational_runs):
            var_output[self.step_name][run_idx] = self.foward_func_map[self.step_name](
                inp
            )
        # turn off dropout for this model
        set_dropout(model=self.model, dropout_flag=False)
        torch.nn.functional.dropout = dropout_off
        var_output = self.stack_variational_outputs(var_output)
        conf_args = {"clean_output": clean_output, "var_output": var_output}
        var_output = self.confidence_func_map[self.step_name](**conf_args)
        return clean_output, var_output
