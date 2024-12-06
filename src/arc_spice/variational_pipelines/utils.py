import copy
import logging
import math
from abc import ABC, abstractmethod
from functools import partial
from typing import Any

import numpy as np
import torch
import transformers
from torch.distributions import Categorical
from torch.nn.functional import softmax
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    ImageToTextPipeline,
    Pipeline,
    TranslationPipeline,
    pipeline,
)

logger = logging.Logger("RTC_variational_pipeline")

# Some methods for the


def collate_scores(
    scores: list[dict[str, float]], label_order
) -> dict[str, list | dict]:
    # these need to be returned in original order
    # return dict for to guarantee class predictions can be recovered
    collated = {score["label"]: score["score"] for score in scores}
    return {
        "scores": [collated[label] for label in label_order],
        "score_dict": collated,
    }


def set_classifier(classifier_pars: dict, device: str) -> transformers.Pipeline:
    # new helper function which given the classifier parameters sets the correct
    # pipeline method. This is needed because they take different kwargs
    # > THIS COULD BE REFACTORED BY PUTTING KWARGS IN THE CONFIG <
    if classifier_pars["specific_task"] == "zero-shot-classification":
        return pipeline(
            task=classifier_pars["specific_task"],
            model=classifier_pars["model"],
            multi_label=True,
            device=device,
            **classifier_pars.get("kwargs", {}),
        )
    return pipeline(
        task=classifier_pars["specific_task"],
        model=classifier_pars["model"],
        device=device,
        **classifier_pars.get("kwargs", {}),
    )


def set_dropout(model: torch.nn.Module, dropout_flag: bool) -> None:
    """
    Turn on or turn off dropout layers of a model.

    Args:
        model: pytorch model
        dropout_flag: dropout -> True/False
    """
    for _, param in model.named_modules():
        if isinstance(param, torch.nn.Dropout):
            # dropout on (True) -> want training mode train(True)
            # dropout off (False) -> eval mode train(False)
            param.train(dropout_flag)


def count_dropout(pipe: Pipeline, dropout_flag: bool) -> int:
    """
    counts the number of dropout layers that are in the configuration that is passed

    Args:
        pipe: pipeline over which to check dropout layers
        dropout_flag: the configuration in which they should be

    Returns:
        dropout_count: The number of layers in the correct configuration
    """
    model = pipe.model
    dropout_count = 0
    for _, param in model.named_modules():
        if isinstance(param, torch.nn.Dropout):
            dropout_count += 1
            assert param.training == dropout_flag

    return dropout_count


# original dropout function
dropout_orig_fn = torch.nn.functional.dropout


def dropout_w_training_override(
    input: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    training_override: bool | None = None,
) -> torch.Tensor:
    """
    Overrides the dropout function to turn it on/off appropriately

    Args:
        ### Dropout function arguments
        input: input tensor
        p: dropout probability. Defaults to 0.5.
        training: training flag. Defaults to True.
        inplace: inplace flag. Defaults to False.
        ### Additional argument
        training_override: Overwrites the training argument to this value.
        Defaults to None.

    Returns:
        Dropout function with override on the training parameter
    """
    if training_override:
        training = training_override

    return dropout_orig_fn(input=input, p=p, training=training, inplace=inplace)


dropout_on = partial(dropout_w_training_override, training_override=True)
dropout_off = partial(dropout_w_training_override, training_override=False)


class RTCVariationalPipelineBase(ABC):
    """
    Base class for the RTC variational pipelines, cannot be instantiated directly, needs
    to have `clean_inference` and `variational_inference` defined by subclass.
    """

    @abstractmethod
    def clean_inference(self, x):
        pass

    @abstractmethod
    def variational_inference(self, x):
        pass

    def __init__(self, zero_shot: bool, n_variational_runs=5, translation_batch_size=8):
        # device for inference
        self.set_device()
        debug_msg_device = f"Loading pipeline on device: {self.device}"
        logging.info(debug_msg_device)
        # map pipeline names to their callable counterparts
        self.func_map = {
            "recognition": self.recognise,
            "translation": self.translate,
            "classification": (
                self.classify_topic_zero_shot if zero_shot else self.classify_topic
            ),
        }
        # the naive outputs of the pipeline stages calculated in self.clean_inference
        self.naive_outputs = {
            "recognition": [
                "outputs",
            ],
            "translation": [
                "full_output",
                "outputs",
                "probs",
                "mean_entropy",
            ],
            "classification": [
                "scores",
            ],
        }
        # parameters for inference process
        self.n_variational_runs = n_variational_runs
        self.translation_batch_size = translation_batch_size

        self.ocr = None
        self.translator = None
        self.classifier = None

        # map pipeline names to their pipeline counterparts
        # to replace class descriptors, we now want class descriptors and the labels
        self.dataset_meta_data: dict = {
            None: None
        }  # This should be defined in subclass if needed

    def _init_pipeline_map(self):
        """
        These need to redefined when overwritten in subclass
        """
        self.pipeline_map = {
            "recognition": self.ocr,
            "translation": self.translator,
            "classification": self.classifier,
        }

    def _init_semantic_density(self):
        """
        Instantiates the NLI tokeniser and model.

        MLNI impl: https://huggingface.co/microsoft/deberta-large-mnli
        """
        self.nli_tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/deberta-large-mnli"
        )

        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-large-mnli"
        )

    def set_device(self):
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    @staticmethod
    def split_translate_inputs(text: str, split_key: str) -> list[str]:
        """
        Splits input text into shorter sequences defined by the `split_key`.

        Args:
            text: input sequence
            split_key: string that the input sequence is split according to.

        Returns:
            recovered_splits: list of split text sequences
        """
        split_rows = text.split(split_key)
        # for when string ends with with the delimiter
        if split_rows[-1] == "":
            split_rows = split_rows[:-1]
        return [split + split_key for split in split_rows]

    @staticmethod
    def check_dropout(pipeline_map: transformers.Pipeline):
        """
        Checks the existence of dropout layers in the models of the pipeline.

        Raises:
            ValueError: Raised when no dropout layers are found.
        """
        logger.debug("\n\n------------------ Testing Dropout --------------------")
        for model_key, pl in pipeline_map.items():
            # only test models that exist
            if pl is None:
                pipeline_none_msg_key = (
                    f"pipeline under model key, `{model_key}`, is currently"
                    " set to None. Was this intended?"
                )
                logger.debug(pipeline_none_msg_key)
                continue
            # turn on dropout for this model
            set_dropout(model=pl.model, dropout_flag=True)
            debug_msg_key = f"Model key: {model_key}"
            logger.debug(debug_msg_key)
            dropout_count = count_dropout(pipe=pl, dropout_flag=True)
            debug_msg_count = (
                f"{dropout_count} dropout layers found in correct configuration."
            )
            logger.debug(debug_msg_count)
            if dropout_count == 0:
                error_message = f"No dropout layers found in {model_key}"
                raise ValueError(error_message)
            set_dropout(model=pl.model, dropout_flag=False)
        logger.debug("-------------------------------------------------------\n\n")

    def recognise(self, inp) -> dict[str, str | list[dict[str, str | torch.Tensor]]]:
        """
        Function to perform OCR.

        Args:
            inp: input dict with key 'ocr_data', containing dict,
                    {
                        'ocr_images': list[ocr images],
                        'ocr_targets': list[ocr target words]
                    }

        Returns:
            dictionary of outputs:
                    {
                        'full_output': [
                            {
                                'generated_text': generated text from ocr model (str),
                                'target': original target text (str)
                            }
                        ],
                        'output': pieced back together string (str)
                    }
        """
        out = self.ocr(inp["ocr_data"]["ocr_images"])  # type: ignore[misc]
        text = " ".join([itm[0]["generated_text"] for itm in out])
        return {
            "full_output": [
                {
                    "target": target,
                    "generated_text": gen_text["generated_text"],
                    "entropies": gen_text["entropies"],
                }
                for target, gen_text in zip(
                    inp["ocr_data"]["ocr_targets"], out, strict=True
                )
            ],
            "output": text,
        }

    def translate(self, text: str) -> dict[str, torch.Tensor | str]:
        """
        Function to perform translation

        Args:
            text: input text

        Returns:
            dictionary with translated text and some other information.
        """
        # split text into sentences
        text_splits = self.split_translate_inputs(text, ".")
        # perform translation on sentences
        translator_outputs = self.translator(  # type: ignore[misc]
            text_splits,
            output_logits=True,
            return_dict_in_generate=True,
            batch_size=self.translation_batch_size,
        )
        # process the outputs, returning just the text components
        sentence_translations = [
            translator_output["translation_text"]
            for translator_output in translator_outputs
        ]
        # join these to create the full translation
        full_translation = ("").join(sentence_translations)
        # record the output and token probabilities
        confidence_metrics = [
            {
                "outputs": translator_output["translation_text"],
                "probs": translator_output["raw_outputs"]["scores"],
                "mean_entropy": torch.mean(translator_output["raw_outputs"]["entropy"])
                .detach()
                .tolist(),
            }
            for translator_output in translator_outputs
        ]
        # group metrics under a single dict key
        stacked_conf_metrics = self.stack_translator_sentence_metrics(
            all_sentence_metrics=confidence_metrics
        )
        # add full output to the output dict
        outputs: dict[str, Any] = {"full_output": full_translation}
        outputs.update(stacked_conf_metrics)
        # {full translation, sentence translations, logits, semantic embeddings}
        return outputs

    def classify_topic(self, text: str) -> dict[str, list[float] | dict]:
        """
        Runs the classification model

        Returns:
            Dictionary of classification outputs, namely the output scores and
            label:score dictionary.
        """
        forward = self.classifier(text, top_k=None)  # type: ignore[misc]
        return collate_scores(forward, self.dataset_meta_data["class_labels"])  # type: ignore[index]

    def classify_topic_zero_shot(self, text: str) -> dict[str, list[float] | dict]:
        """
        Runs the zero-shot classification model

        Returns:
            Dictionary of classification outputs, namely the output scores and
            label:score dictionary.
        """
        labels = [
            descriptors["en"]
            for descriptors in self.dataset_meta_data["class_descriptors"]  # type: ignore[index]
        ]
        forward = self.classifier(text, labels)  # type: ignore[misc]
        return collate_scores(
            [
                {"label": label, "score": score}
                for label, score in zip(
                    forward["labels"], forward["scores"], strict=True
                )
            ],
            label_order=labels,
        )

    def stack_translator_sentence_metrics(
        self, all_sentence_metrics: list[dict[str, Any]]
    ) -> dict[str, list[Any]]:
        """
        Stacks values from dictionary list into lists under a single key

        Returns:
            Dictionary with each metric under a single key
        """
        stacked = {}
        # we skip the full output
        for metric in self.naive_outputs["translation"][1:]:
            stacked[metric] = [
                sentence_metrics[metric] for sentence_metrics in all_sentence_metrics
            ]
        return stacked

    def stack_variational_outputs(
        self, var_output: dict[str, list[dict]]
    ) -> dict[str, list]:
        """
        Similar to above but this stacks variational output dictinaries into lists
        under a single key.
        """
        # Create new dict
        new_var_dict: dict[str, Any] = {}
        # For each key create a new dict
        for step in var_output:
            new_var_dict[step] = {}
            # for each metric in a clean inference run (naive_ouputs)
            for metric in self.naive_outputs[step]:
                # create a list for each item
                new_values = [None] * self.n_variational_runs
                # populate list with items and assign it in new dictionary
                for run in range(self.n_variational_runs):
                    new_values[run] = var_output[step][run][metric]
                new_var_dict[step][metric] = new_values
        # overwrite the existing output dictionary
        return new_var_dict

    def sentence_density(
        self,
        clean_sentence: str,
        var_sentences: list[str],
        var_scores: list[torch.Tensor],
    ) -> tuple[float, int]:
        """
        Caclulates the semantic density of a given sentence, using the clean and
        variational outputs

        Args:
            clean_sentence: clean inference output on sentence
            var_sentences: variational outputs on the sentence
            var_scores: variational output probabilities

        Returns:
            semantic_density: semantic density measurement
            sequence_length: length of the sequence (for overall weighting)
        """
        # calc sequence lengths (for sentence weighting)
        sequence_length = len(
            self.nli_tokenizer.encode(
                clean_sentence, padding=True, return_tensors="pt"
            )[0]
        )
        # define list of the kernel functions and conditional probabilities
        kernel_funcs = torch.zeros(self.n_variational_runs)
        cond_probs = torch.zeros(self.n_variational_runs)
        # for each variational run of the sentence
        for var_index, var_sentence in enumerate(var_sentences):
            # run the nli model
            nli_inp = "[CLS]" + clean_sentence + " [SEP] " + var_sentence + "[SEP]"
            encoded_nli = self.nli_tokenizer.encode(
                nli_inp, padding=True, return_tensors="pt"
            )
            nli_out = softmax(self.nli_model(encoded_nli)["logits"], dim=-1)[0]
            contradiction = nli_out[0]
            neutral = nli_out[1]
            # calculate kernel function: from https://arxiv.org/pdf/2405.13845
            kernel_funcs[var_index] = 1 - (contradiction + (0.5 * neutral))

        # TODO vectorize
        # calculate conditional probabilities take power first to avoid NaN
        for var_index, var_score_out in enumerate(var_scores):
            var_score = var_score_out.squeeze()
            cond_probs[var_index] = torch.prod(
                torch.pow(var_score, 1 / len(var_score)), dim=-1
            )
        # caclulate semantic density measure
        semantic_density = (1 / torch.sum(cond_probs)) * torch.sum(
            torch.mul(cond_probs, kernel_funcs)
        )
        return semantic_density.item(), sequence_length

    def translation_semantic_density(
        self, clean_output: dict, var_output: dict, **kwargs
    ) -> dict[str, float | Any]:
        """
        Runs the semantic density measurement from https://arxiv.org/pdf/2405.13845.

        ### Broadly:

        For each sentence in translation:
            Use NLI model to determine semantic similarity to clean sentence
            (1 - (contradiction + 0.5 * neutral))

        Take mean, weighted by length of sentence

        Need to loop over the sentences and calculate metrics comparing against clean
        Average these, weighted by the sequence length giving the overall confidence.
        """
        # get the clean and variational output and record number of sentences
        clean_out = clean_output["translation"]["outputs"]
        var_steps = var_output["translation"]
        n_sentences = len(clean_out)
        # define empty lists for the measurements
        densities: list[Any] = [None] * n_sentences
        sequence_lengths: list[Any] = [None] * n_sentences
        # stack the variational runs according to their sentences, then loop and pass to
        # density calculation function
        for sentence_index, clean_sentence in enumerate(clean_out):
            var_sentences = [step[sentence_index] for step in var_steps["outputs"]]
            var_probs = [step[sentence_index] for step in var_steps["probs"]]
            density, seqlen = self.sentence_density(
                clean_sentence=clean_sentence,
                var_sentences=var_sentences,
                var_scores=var_probs,
            )
            # assign to our output list
            densities[sentence_index] = density
            sequence_lengths[sentence_index] = seqlen

        # perform weighting
        total_len = torch.sum(torch.tensor(sequence_lengths))
        weighted_average = (
            torch.sum(torch.tensor(densities) * torch.tensor(sequence_lengths))
            / total_len
        )
        # update the output dict
        var_output["translation"].update(
            {
                "semantic_densities": densities,
                "weighted_semantic_density": weighted_average.item(),
                "sequence_length": sequence_lengths,
            }
        )

        return var_output

    def get_classification_confidence(
        self, var_output: dict, epsilon: float = 1e-15, **kwargs
    ) -> dict[str, float | torch.Tensor]:
        """
        _summary_

        Args:
            var_output: the variational_input on which to act
            epsilon: _description_. Defaults to 1e-15.
        """
        # From: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9761166
        # stack probabilities into single tensor
        stacked_probs = torch.stack(
            [torch.tensor(pred) for pred in var_output["classification"]["scores"]],
            dim=1,
        )
        # calculate mean predictions
        means = torch.mean(stacked_probs, dim=-1)
        # predictive entropy for each class
        pred_entropies = -1 * (
            means * torch.log(means + epsilon)
            + (1 - means) * torch.log((1 - means) + epsilon)
        )
        # individual entropies
        all_entropies = -1 * (
            stacked_probs * torch.log(stacked_probs + epsilon)
            + (1 - stacked_probs) * torch.log((1 - stacked_probs) + epsilon)
        )
        # mutual information is difference between the predicted entropy and mean of the
        # entropies
        mutual_info = pred_entropies - torch.mean(all_entropies, dim=-1)
        # other measures
        stds = torch.std(stacked_probs, dim=-1)
        variances = torch.var(stacked_probs, dim=-1)
        # return all metrics
        var_output["classification"].update(
            {
                "mean_scores": means,
                "std_scores": stds,
                "var_scores": variances,
                "predicted_entropy": pred_entropies,
                "mutual_information": mutual_info,
            }
        )
        return var_output

    def get_ocr_confidence(self, var_output: dict) -> dict[str, float]:
        """Generate the ocr confidence score.

        Args:
            var_output: variational run outputs

        Returns:
            dictionary with metrics
        """
        # Adapted for variational methods from: https://arxiv.org/pdf/2412.01221
        stacked_entropies = torch.stack(
            [
                [data["entropies"] for data in output["full_output"]]
                for output in var_output["recognition"]
            ],
            dim=1,
        )
        # mean entropy
        mean = torch.mean(stacked_entropies)
        var_output["recognition"].update({"mean_entropy": mean})
        return var_output


# Translation pipeline with additional functionality to save logits from fwd pass
class CustomTranslationPipeline(TranslationPipeline):
    """
    custom translation pipeline to return the logits with the generated text. Largely
    the same as the pytorch version with some additional arguments passed to the
    `generate` method.
    """

    def postprocess(
        self,
        model_outputs: dict,
        **postprocess_params,
    ):
        # model_outputs gets overwritten in the super().postprocess call
        # make a copy here so we retain the information we want
        raw_out = copy.deepcopy(model_outputs)
        processed = super().postprocess(model_outputs, **postprocess_params)

        return {
            "translation_text": processed[0]["translation_text"],
            "raw_outputs": raw_out,
        }

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

        # logits are a tuple of length output_ids[-1]-1
        # each element is a tensor of shape (batch_size, vocab_size)
        logits = torch.stack(out["logits"], dim=1)
        # get softmax of the logits to get token probabilities
        softmax_logits = softmax(logits, dim=-1)
        vocab_size = softmax_logits.shape[-1]
        normalised_entropy = torch.distributions.Categorical(
            probs=softmax_logits
        ).entropy() / math.log(vocab_size)
        max_token_scores = torch.max(softmax_logits, dim=-1).values

        return {
            "output_ids": output_ids,
            "scores": max_token_scores,
            "entropy": normalised_entropy,
        }


class CustomOCRPipeline(ImageToTextPipeline):
    """
    custom OCR pipeline to return logits with the generated text.
    """

    def postprocess(self, model_outputs: dict, **postprocess_params):
        raw_out = copy.deepcopy(model_outputs)
        processed = super().postprocess(
            model_outputs["model_output"], **postprocess_params
        )

        return {"generated_text": processed[0]["generated_text"], "raw_output": raw_out}

    def _forward(self, model_inputs, **generate_kwargs):
        if (
            "input_ids" in model_inputs
            and isinstance(model_inputs["input_ids"], list)
            and all(x is None for x in model_inputs["input_ids"])
        ):
            model_inputs["input_ids"] = None

        inputs = model_inputs.pop(self.model.main_input_name)
        out = self.model.generate(
            inputs,
            **model_inputs,
            **generate_kwargs,
            output_logits=True,
            return_dict_in_generate=True,
        )

        logits = torch.stack(out.logits, dim=1)
        entropy = Categorical(logits=logits).entropy() / np.log(logits[0].size()[1])
        return {"model_output": out.sequences, "entropies": entropy}
