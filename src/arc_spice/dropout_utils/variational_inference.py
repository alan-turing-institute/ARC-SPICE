import copy
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.functional import cosine_similarity, softmax
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    AutoModel,
    AutoModelForSequenceClassification,
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

    def __init__(self, pars: dict[str : dict[str:str]], n_variational_runs=5):
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

        self.nli_tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/deberta-large-mnli"
        )

        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-large-mnli"
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
        self.naive_outputs = {
            "outputs",
            "logits",
            "entropy",
            "normalised_entropy",
            "probs",
            "semantic_embedding",
        }
        self.n_variational_runs = n_variational_runs

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

    def collect_metrics(self):
        new_var_dict = {}
        for step in self.var_output["variational"].keys():
            new_var_dict[step] = {}
            for metric in self.naive_outputs:
                new_values = [None] * self.n_variational_runs
                for run in range(self.n_variational_runs):
                    new_values[run] = self.var_output["variational"][step][run][metric]
                new_var_dict[step][metric] = new_values

        self.var_output["variational"] = new_var_dict

    def calculate_semantic_density(self):
        for step in self.var_output["variational"].keys():
            clean_out = self.var_output["clean"][step]["outputs"]
            var_step = self.var_output["variational"][step]
            kernel_funcs = torch.zeros(self.n_variational_runs)
            cond_probs = torch.zeros(self.n_variational_runs)
            sims = [None] * self.n_variational_runs
            for run_index, run_out in enumerate(var_step["outputs"]):
                run_prob = var_step["probs"][run_index]
                nli_inp = clean_out + " [SEP] " + run_out
                encoded_nli = self.nli_tokenizer.encode(
                    nli_inp, padding=True, return_tensors="pt"
                )
                sims[run_index] = cosine_similarity(
                    self.var_output["clean"][step]["semantic_embedding"],
                    var_step["semantic_embedding"][run_index],
                )
                nli_out = softmax(self.nli_model(encoded_nli)["logits"], dim=-1)[0]
                kernel_func = 1 - (nli_out[0] + (0.5 * nli_out[1]))
                cond_probs[run_index] = torch.pow(
                    torch.prod(run_prob, -1), 1 / len(run_prob)
                )
                kernel_funcs[run_index] = kernel_func
            semantic_density = (
                1
                / (torch.sum(cond_probs))
                * torch.sum(torch.mul(cond_probs, kernel_funcs))
            )
            self.var_output["variational"][step].update(
                {"semantic_density": semantic_density.item(), "cosine_similarity": sims}
            )

    def clean_inference(self, x: Union[np.ndarray, bytes, str]):
        """

        Run the pipeline on an input x

        Args:
            x: numpy array audio input

        Returns:
            summarised transcript with associated unvertainties at each step
        """

        self.clean_output = {step: {} for step in self.pipeline_map.keys()}
        # transcription
        transcription = self.transcribe(x)
        self.clean_output["transcription"].update(transcription)

        # translation
        translation = self.translate(transcription["outputs"])
        self.clean_output["translation"].update(translation)

        # summarisation
        summarisation = self.summarise(translation["outputs"])
        self.clean_output["summarisation"].update(summarisation)

    def variational_inference(self, x):
        # we need clean inputs to pass to each step, we run that first
        self.var_output = {"clean": {}, "variational": {}}
        self.clean_inference(x)
        self.var_output["clean"] = self.clean_output
        # each step accepts a different input from the clean pipeline
        input_map = {
            "transcription": x,
            "translation": self.var_output["clean"]["transcription"]["outputs"],
            "summarisation": self.var_output["clean"]["translation"]["outputs"],
        }
        # for each model in pipeline
        for model_key, pl in self.pipeline_map.items():
            # turn on dropout for this model
            set_dropout(model=pl.model, dropout_flag=True)
            # create the output list
            self.var_output["variational"][model_key] = [None] * self.n_variational_runs
            # do n runs of the inference
            for run_idx in range(self.n_variational_runs):
                self.var_output["variational"][model_key][run_idx] = self.func_map[
                    model_key
                ](input_map[model_key])
            # turn off dropout for this model
            set_dropout(model=pl.model, dropout_flag=False)

        self.collect_metrics()
        self.calculate_semantic_density()

    def __call__(self, x):
        self.clean_inference(x)
        return self.clean_output


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
