import copy
from typing import Union

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn.functional import softmax
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    SummarizationPipeline,
    TranslationPipeline,
    pipeline,
)

from arc_spice.dropout_utils.dropout_pipeline import set_dropout, test_dropout


def get_confidence_metrics(logits: torch.Tensor) -> dict[str : torch.Tensor]:
    """
    calculates confidence metrics for a tensor of logits:
    - entropy : token-wise entropy
    - normalised entropy : token-wise entropy normalised by vocab size
    - probs : log-probabilities of the each generated token

    Returns:
        dictionary containing the calculated confidence metrics
    """
    vocab = torch.tensor(logits.shape[-1])
    entropy = Categorical(logits=logits).entropy()
    normalised_entropy = entropy / torch.log(vocab)
    softmax_logits = softmax(logits, dim=-1)
    max_probs = torch.max(softmax_logits, dim=-1).values
    return {
        "entropy": entropy,
        "normalised_entropy": normalised_entropy,
        "probs": max_probs,
    }


class TTSVariationalPipeline:
    """
    variational version of the TTS pipeline
    """

    def __init__(self, pars: dict[str : dict[str:str]]):
        self.transcriber = pipeline(
            task=pars["transcriber"]["specific_task"],
            model=pars["transcriber"]["model"],
            pipeline_class=CustomSpeechRecognitionPipeline,
        )
        self.translator = pipeline(
            task=pars["translator"]["specific_task"],
            model=pars["translator"]["model"],
            max_length=1024,
            pipeline_class=CustomTranslationPipeline,
        )
        self.summariser = pipeline(
            task=pars["summariser"]["specific_task"],
            model=pars["summariser"]["model"],
            pipeline_class=CustomSummarizationPipeline,
        )

        self.pipeline_map = {
            "transcription": self.transcriber,
            "translation": self.translator,
            "summarisation": self.summariser,
        }
        self.generate_kwargs = {"output_scores": True}

    def collect_confidence_metrics(
        self, output: dict[str : dict[str : torch.Tensor]]
    ) -> dict[str : dict[str : torch.Tensor]]:
        """
        For each step/model in the pipeline calculates the associated uncertainty
        metrics using the logits

        Args:
            output: dictionary containing the outputs of each step

        Returns:
            updated dictionary containing the confidence metrics calculated for each
            step in the pipeline
        """
        for step in self.pipeline_map.keys():
            output[step].update(get_confidence_metrics(output[step]["logits"]))
        return output

    def __call__(self, x: Union[np.ndarray, bytes, str]):
        """

        Run the pipeline on an input x

        Args:
            x: numpy array audio input

        Returns:
            summarised transcript with associated unvertainties at each step
        """

        output = {step: {} for step in self.pipeline_map.keys()}
        # transcription
        transcription = self.transcriber(x, generate_kwargs=self.generate_kwargs)
        output["transcription"]["outputs"] = transcription["text"]
        output["transcription"]["logits"] = (
            transcription["raw_outputs"][0]["logits"].squeeze().T
        )
        # translation
        translation = self.translator(
            transcription["text"],
            output_logits=True,
            return_dict_in_generate=True,
        )
        output["translation"]["outputs"] = translation["translation_text"]
        output["translation"]["logits"] = torch.cat(
            translation["raw_outputs"]["logits"]
        )
        # summarisation
        summarisation = self.summariser(
            translation["translation_text"],
            output_logits=True,
            return_dict_in_generate=True,
        )
        output["summarisation"]["outputs"] = summarisation["summary_text"]
        output["summarisation"]["logits"] = torch.cat(
            summarisation["raw_outputs"]["logits"]
        )

        # add confidence metrics using the logits
        output = self.collect_confidence_metrics(output=output)

        return output


class CustomSpeechRecognitionPipeline(AutomaticSpeechRecognitionPipeline):
    def postprocess(
        self,
        model_outputs: dict,
        **postprocess_params,
    ):
        # model_outputs gets overwritten in the super().postprocess call
        # make a copy here so we retain the information we want
        raw_out = copy.deepcopy(model_outputs)
        processed = super().postprocess(model_outputs, **postprocess_params)

        new_output = {"text": processed["text"], "raw_outputs": raw_out}
        return new_output


class CustomTranslationPipeline(TranslationPipeline):
    def postprocess(
        self,
        model_outputs: dict,
        **postprocess_params,
    ):
        # model_outputs gets overwritten in the super().postprocess call
        # make a copy here so we retain the information we want
        raw_out = copy.deepcopy(model_outputs)
        processed = super().postprocess(model_outputs, **postprocess_params)

        new_output = {
            "translation_text": processed[0]["translation_text"],
            "raw_outputs": raw_out,
        }
        return new_output

    def _forward(self, model_inputs, **generate_kwargs):
        if self.framework == "pt":
            in_b, input_length = model_inputs["input_ids"].shape
        elif self.framework == "tf":
            raise NotImplementedError

        self.check_inputs(
            input_length,
            generate_kwargs.get("min_length", self.model.config.min_length),
            generate_kwargs.get("max_length", self.model.config.max_length),
        )
        out = self.model.generate(**model_inputs, **generate_kwargs)
        output_ids = out["sequences"]
        out_b = output_ids.shape[0]
        if self.framework == "pt":
            output_ids = output_ids.reshape(in_b, out_b // in_b, *output_ids.shape[1:])
        elif self.framework == "tf":
            raise NotImplementedError
        return {"output_ids": output_ids, "logits": out["logits"]}


class CustomSummarizationPipeline(SummarizationPipeline):

    def postprocess(
        self,
        model_outputs: dict,
        **postprocess_params,
    ):
        # model_outputs gets overwritten in the super().postprocess call
        # make a copy here so we retain the information we want
        raw_out = copy.deepcopy(model_outputs)
        processed = super().postprocess(model_outputs, **postprocess_params)

        new_output = {
            "summary_text": processed[0]["summary_text"],
            "raw_outputs": raw_out,
        }
        return new_output

    def _forward(self, model_inputs, **generate_kwargs):
        if self.framework == "pt":
            in_b, input_length = model_inputs["input_ids"].shape
        elif self.framework == "tf":
            raise NotImplementedError

        self.check_inputs(
            input_length,
            generate_kwargs.get("min_length", self.model.config.min_length),
            generate_kwargs.get("max_length", self.model.config.max_length),
        )
        out = self.model.generate(**model_inputs, **generate_kwargs)
        output_ids = out["sequences"]
        out_b = output_ids.shape[0]
        if self.framework == "pt":
            output_ids = output_ids.reshape(in_b, out_b // in_b, *output_ids.shape[1:])
        elif self.framework == "tf":
            raise NotImplementedError
        return {"output_ids": output_ids, "logits": out["logits"]}
