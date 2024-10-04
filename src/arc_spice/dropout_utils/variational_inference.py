import copy
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.functional import softmax
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    AutoModel,
    AutoTokenizer,
    SummarizationPipeline,
    TranslationPipeline,
    pipeline,
)

from arc_spice.dropout_utils.dropout_pipeline import set_dropout

# From huggingface page with model:
#   - https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2


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

        self.semantic_tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.semantic_model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.pipeline_map = {
            "transcription": self.transcriber,
            "translation": self.translator,
            "summarisation": self.summariser,
        }
        self.generate_kwargs = {"output_scores": True}
        self.func_map = {
            "transcription": self.transcribe,
            "translation": self.translate,
            "summarisation": self.summarise,
        }

    def get_confidence_metrics(
        self, output_dict: dict[str : str | torch.Tensor]
    ) -> dict[str : torch.Tensor]:
        """
        calculates confidence metrics for a tensor of logits:
        - entropy : token-wise entropy
        - normalised entropy : token-wise entropy normalised by vocab size
        - probs : log-probabilities of the each generated token

        Returns:
            dictionary containing the calculated confidence metrics
        """
        logits = output_dict["logits"]
        text = output_dict["outputs"]
        vocab = torch.tensor(logits.shape[-1])
        entropy = Categorical(logits=logits).entropy()
        normalised_entropy = entropy / torch.log(vocab)
        softmax_logits = softmax(logits, dim=-1)
        max_probs = torch.max(softmax_logits, dim=-1).values
        tokenized_text = self.semantic_tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            model_embeddings = self.semantic_model(**tokenized_text)
        # Perform pooling
        sentence_embeddings = mean_pooling(
            model_embeddings, tokenized_text["attention_mask"]
        )

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return {
            "entropy": entropy,
            "normalised_entropy": normalised_entropy,
            "probs": max_probs,
            "semantic_embedding": sentence_embeddings,
        }

    def transcribe(self, x: Union[np.ndarray, bytes, str]):
        transcription = self.transcriber(x, generate_kwargs=self.generate_kwargs)
        output_text = transcription["text"]
        output_logits = transcription["raw_outputs"][0]["logits"].squeeze().T
        output_dict = {"outputs": output_text, "logits": output_logits}
        confidence_metrics = self.get_confidence_metrics(output_dict)
        output_dict.update(confidence_metrics)
        return output_dict

    def translate(self, source_text: str):
        translation = self.translator(
            source_text,
            output_logits=True,
            return_dict_in_generate=True,
        )
        output_text = translation["translation_text"]
        output_logits = torch.cat(translation["raw_outputs"]["logits"])
        output_dict = {"outputs": output_text, "logits": output_logits}
        confidence_metrics = self.get_confidence_metrics(output_dict)
        output_dict.update(confidence_metrics)
        return output_dict

    def summarise(self, source_text: str):
        summarisation = self.summariser(
            source_text,
            output_logits=True,
            return_dict_in_generate=True,
        )
        output_text = summarisation["summary_text"]
        output_logits = torch.cat(summarisation["raw_outputs"]["logits"])
        output_dict = {"outputs": output_text, "logits": output_logits}
        confidence_metrics = self.get_confidence_metrics(output_dict)
        output_dict.update(confidence_metrics)
        return output_dict

    def clean_inference(self, x: Union[np.ndarray, bytes, str]):
        """

        Run the pipeline on an input x

        Args:
            x: numpy array audio input

        Returns:
            summarised transcript with associated unvertainties at each step
        """

        output = {step: {} for step in self.pipeline_map.keys()}
        # transcription
        transcription = self.transcribe(x)
        output["transcription"].update(transcription)

        # translation
        translation = self.translate(transcription["outputs"])
        output["translation"].update(translation)

        # summarisation
        summarisation = self.summarise(translation["outputs"])
        output["summarisation"].update(summarisation)

        return output

    def variational_inference(self, x, n_runs=5):
        # we need clean inputs to pass to each step, we run that first
        output = {"clean": {}, "variational": {}}
        output["clean"] = self.clean_inference(x)
        # each step accepts a different input from the clean pipeline
        input_map = {
            "transcription": x,
            "translation": output["clean"]["transcription"]["outputs"],
            "summarisation": output["clean"]["translation"]["outputs"],
        }
        # for each model in pipeline
        for model_key, pl in self.pipeline_map.items():
            # turn on dropout for this model
            set_dropout(model=pl.model, dropout_flag=True)
            # create the output list
            output["variational"][model_key] = [None] * n_runs
            # do n runs of the inference
            for run_idx in range(n_runs):
                output["variational"][model_key][run_idx] = self.func_map[model_key](
                    input_map[model_key]
                )
            # turn off dropout for this model
            set_dropout(model=pl.model, dropout_flag=False)
        return output

    def __call__(self, x):
        return self.clean_inference(x)


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
