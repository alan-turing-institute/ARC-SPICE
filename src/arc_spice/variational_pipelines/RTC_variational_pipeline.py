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

from arc_spice.variational_pipelines.dropout_utils import set_dropout

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


def identity(x):
    return x


class RTCVariationalPipeline:
    """
    variational version of the RTC pipeline
    """

    def __init__(
        self, model_pars: dict[str : dict[str:str]], data_pars, n_variational_runs=5
    ) -> None:
        self.OCR = pipeline(
            task=model_pars["OCR"]["specific_task"],
            model=model_pars["OCR"]["model"],
        )
        self.translator = pipeline(
            task=model_pars["translator"]["specific_task"],
            model=model_pars["translator"]["model"],
            max_length=512,
            pipeline_class=CustomTranslationPipeline,
        )

        self.classifier = pipeline(
            task=model_pars["classifier"]["specific_task"],
            model=model_pars["classifier"]["model"],
            multi_label=True,
        )

        self.candidate_labels = [
            class_names_dict["en"]
            for class_names_dict in data_pars["class_descriptors"]
        ]

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
            "recognition": self.OCR,
            "translation": self.translator,
            "classification": self.classifier,
        }
        self.generate_kwargs = {"output_scores": True}

        self.func_map = {
            "recognition": self.recognise,
            "translation": self.translate,
            "classification": self.classify,
        }
        self.naive_outputs = {
            # "recognition": [
            #     "outputs",
            #     "logits",
            #     "entropy",
            #     "normalised_entropy",
            #     "probs",
            #     "semantic_embedding",
            # ],
            "recognition": [
                "outputs",
                # "logits",
                # "entropy",
                # "normalised_entropy",
                # "probs",
                # "semantic_embedding",
            ],
            "translation": [
                "outputs",
                "logits",
                "entropy",
                "normalised_entropy",
                "probs",
                "semantic_embedding",
            ],
            # "classification": [
            #     "outputs",
            #     "logits",
            #     "entropy",
            #     "normalised_entropy",
            #     "probs",
            # ],
            "classification": [
                # "outputs",
                # "logits",
                # "entropy",
                # "normalised_entropy",
                "probs",
            ],
        }
        self.n_variational_runs = n_variational_runs

    @staticmethod
    def split_inputs(text, split_key):
        split_rows = text.split(split_key)
        recovered_splits = [split + split_key for split in split_rows]
        return recovered_splits

    def get_sentence_conf(self, sentence):
        output_logits = sentence["raw_outputs"]["logits"].squeeze()
        vocab = torch.tensor(output_logits.shape[-1])
        entropy = Categorical(logits=output_logits).entropy()
        normalised_entropy = entropy / torch.log(vocab)
        softmax_logits = softmax(output_logits, dim=-1)
        max_probs = torch.max(softmax_logits, dim=-1).values

        tokenized_text = self.semantic_tokenizer(
            sentence["translation_text"],
            padding=True,
            truncation=True,
            return_tensors="pt",
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
            "outputs": sentence["translation_text"],
            "logits": output_logits,
            "entropy": entropy,
            "normalised_entropy": normalised_entropy,
            "probs": max_probs,
            "semantic_embedding": sentence_embeddings,
        }

    def recognise(self, inp):
        return {"outputs": inp}

    def translate(self, text):
        text_splits = self.split_inputs(text, ".")

        translator_outputs = self.translator(
            text_splits, output_logits=True, return_dict_in_generate=True, batch_size=8
        )
        sentence_translations = [
            translator_output["translation_text"]
            for translator_output in translator_outputs
        ]
        full_translation = ("").join(sentence_translations)
        confidence_metrics = [
            self.get_sentence_conf(translator_output)
            for translator_output in translator_outputs
        ]
        stacked_conf_metrics = self.stack_translator_sentence_metrics(
            confidence_metrics
        )
        outputs = {"full_output": full_translation}
        outputs.update(stacked_conf_metrics)
        return outputs

    def classify(self, text):
        forward = self.classifier(text, self.candidate_labels)
        return {"probs": forward["scores"]}

    def stack_translator_sentence_metrics(self, all_sentence_metrics):
        stacked = {}
        for metric in self.naive_outputs["translation"]:
            stacked[metric] = [
                sentence_metrics[metric] for sentence_metrics in all_sentence_metrics
            ]
        return stacked

    def stack_variational_outputs(self):
        new_var_dict = {}
        print(self.var_output["variational"].keys())
        for step in self.var_output["variational"].keys():
            print(step)
            new_var_dict[step] = {}
            for metric in self.naive_outputs[step]:
                new_values = [None] * self.n_variational_runs
                for run in range(self.n_variational_runs):
                    new_values[run] = self.var_output["variational"][step][run][metric]
                new_var_dict[step][metric] = new_values

        self.var_output["variational"] = new_var_dict

    def clean_inference(self, x):
        self.clean_output = {
            "recognition": {},
            "translation": {},
            "classification": {},
        }
        """Run the pipeline on an input x"""

        # UNTIL THE OCR DATA IS AVAILABLE
        # recognition = self.OCR(x)
        # self.results["OCR"] = recognition["text"]
        self.clean_output["recognition"] = self.recognise(x)
        # UNTIL THE OCR DATA IS AVAILABLE

        self.clean_output["translation"] = self.translate(
            self.clean_output["recognition"]["outputs"]
        )
        self.clean_output["classification"] = self.classify(
            self.clean_output["translation"]["outputs"]
        )

    def variational_inference(self, x):
        self.clean_inference(x)
        self.var_output = {"clean": self.clean_output, "variational": {}}

        input_map = {
            "recognition": x,
            "translation": self.clean_output["recognition"]["outputs"],
            "classification": self.clean_output["translation"]["full_output"],
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

        self.stack_variational_outputs()

    def __call__(self, x):
        self.clean_inference(x)
        return self.clean_output


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
