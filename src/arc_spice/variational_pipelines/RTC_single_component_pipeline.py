from typing import Any

import torch
from transformers import pipeline

from arc_spice.variational_pipelines.RTC_variational_pipeline import (
    CustomOCRPipeline,
    CustomTranslationPipeline,
    RTCVariationalPipelineBase,
)
from arc_spice.variational_pipelines.utils import dropout_off, dropout_on, set_dropout


class RTCSingleComponentPipeline(RTCVariationalPipelineBase):
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
        step_name,
        input_key,
        forward_function,
        confidence_function,
        n_variational_runs=5,
        translation_batch_size=8,
    ) -> None:
        super().__init__(n_variational_runs, translation_batch_size)
        # define objects that are needed and nothing else
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
        self.step_name = step_name
        self.input_key = input_key
        self.forward_function = forward_function
        self.confidence_function = confidence_function

    # clean inference
    def clean_inference(self, x):
        # run only the model that is defined
        inp = x[self.input_key]
        clean_output: dict[str, Any] = {
            self.step_name: {},
        }
        clean_output[self.step_name] = self.forward_function(inp)
        return clean_output

    def variational_inference(self, x):
        # run only model that is defined in clean output
        clean_output = self.clean_inference(x)
        inp = x[self.input_key]
        var_output: dict[str, Any] = {
            self.step_name: {},
        }
        # variational stage is the same as the full pipeline model, with different input
        # turn on dropout for this model
        # model will be defined in the subclass
        set_dropout(model=self.model, dropout_flag=True)  # type: ignore[attr-defined]
        torch.nn.functional.dropout = dropout_on
        # do n runs of the inference
        for run_idx in range(self.n_variational_runs):
            var_output[self.step_name][run_idx] = self.forward_function(inp)
        # turn off dropout for this model
        set_dropout(model=self.model, dropout_flag=False)  # type: ignore[attr-defined]
        torch.nn.functional.dropout = dropout_off
        var_output = self.stack_variational_outputs(var_output)
        # For confidence function we need to pass both outputs in all cases
        # This allows the abstraction to self.confidence_func_map[self.step_name]
        conf_args = {"clean_output": clean_output, "var_output": var_output}
        var_output = self.confidence_function(**conf_args)
        # return both as in base function method
        return clean_output, var_output


class RecognitionVariationalPipeline(RTCSingleComponentPipeline):
    def __init__(
        self,
        model_pars: dict[str, dict[str, str]],
        n_variational_runs=5,
        ocr_batch_size=64,
        **kwargs,
    ):
        self.set_device()
        self.ocr = pipeline(
            model=model_pars["OCR"]["model"],
            device=self.device,
            pipeline_class=CustomOCRPipeline,
            max_new_tokens=20,
            batch_size=ocr_batch_size,
            **kwargs,
        )
        self.model = self.ocr.model
        super().__init__(
            step_name="recognition",
            input_key="ocr_data",
            forward_function=self.recognise,
            confidence_function=self.get_ocr_confidence,
            n_variational_runs=n_variational_runs,
            **kwargs,
        )
        self._init_pipeline_map()


class TranslationVariationalPipeline(RTCSingleComponentPipeline):
    def __init__(
        self,
        model_pars: dict[str, dict[str, str]],
        n_variational_runs=5,
        translation_batch_size=8,
        **kwargs,
    ):
        self.set_device()
        # need to initialise the NLI models in this case
        self._init_semantic_density()
        super().__init__(
            step_name="translation",
            input_key="source_text",
            forward_function=self.translate,
            confidence_function=self.translation_semantic_density,
            n_variational_runs=n_variational_runs,
            translation_batch_size=translation_batch_size,
        )
        self.translator = pipeline(
            task=model_pars["translator"]["specific_task"],
            model=model_pars["translator"]["model"],
            max_length=512,
            pipeline_class=CustomTranslationPipeline,
            device=self.device,
        )
        self.model = self.translator.model
        self._init_pipeline_map()


class ClassificationVariationalPipeline(RTCSingleComponentPipeline):
    """
    Classification Pipeline

    Args:
        RTCSingleComponentPipeline: Subclass of the `SingleComponentPipeline` base class
    """

    def __init__(
        self,
        model_pars: dict[str, dict[str, str]],
        data_pars: dict[str, Any],
        n_variational_runs=5,
        **kwargs,
    ):
        self.set_device()
        super().__init__(
            step_name="classification",
            input_key="target_text",
            forward_function=self.classify_topic,
            confidence_function=self.get_classification_confidence,
            n_variational_runs=n_variational_runs,
            **kwargs,
        )
        self.classifier = pipeline(
            task=model_pars["classifier"]["specific_task"],
            model=model_pars["classifier"]["model"],
            multi_label=True,
            device=self.device,
        )
        self.model = self.classifier.model
        # topic description labels for the classifier
        self.topic_labels = [
            class_names_dict["en"]
            for class_names_dict in data_pars["class_descriptors"]
        ]
        self._init_pipeline_map()
