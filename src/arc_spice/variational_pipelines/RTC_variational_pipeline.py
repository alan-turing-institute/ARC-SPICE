import copy
import logging
from typing import Any

import torch
from torch.nn.functional import softmax
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TranslationPipeline,
    pipeline,
)

from arc_spice.variational_pipelines.dropout_utils import (
    count_dropout,
    dropout_off,
    dropout_on,
    set_dropout,
)

logger = logging.Logger("RTC_variational_pipeline")


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
class RTCVariationalPipeline:
    """
    variational version of the RTC pipeline
    """

    def __init__(
        self,
        model_pars: dict[str, dict[str, str]],
        data_pars,
        n_variational_runs=5,
        translation_batch_size=8,
    ) -> None:
        # device for inference
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        debug_msg_device = f"Loading pipeline on device: {device}"
        logging.info(debug_msg_device)

        # defining the pipeline objects
        self.ocr = pipeline(
            task=model_pars["OCR"]["specific_task"],
            model=model_pars["OCR"]["model"],
            device=device,
        )
        self.translator = pipeline(
            task=model_pars["translator"]["specific_task"],
            model=model_pars["translator"]["model"],
            max_length=512,
            pipeline_class=CustomTranslationPipeline,
            device=device,
        )
        self.classifier = pipeline(
            task=model_pars["classifier"]["specific_task"],
            model=model_pars["classifier"]["model"],
            multi_label=True,
            device=device,
        )
        # topic description labels for the classifier
        self.topic_labels = [
            class_names_dict["en"]
            for class_names_dict in data_pars["class_descriptors"]
        ]

        self._init_semantic_density()

        # map pipeline names to their pipeline counterparts
        self.pipeline_map = {
            "recognition": self.ocr,
            "translation": self.translator,
            "classification": self.classifier,
        }
        # map pipeline names to their callable counterparts
        self.func_map = {
            "recognition": self.recognise,
            "translation": self.translate,
            "classification": self.classify_topic,
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
            ],
            "classification": [
                "scores",
            ],
        }
        # parameters for inference process
        self.n_variational_runs = n_variational_runs
        self.translation_batch_size = translation_batch_size

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

    def check_dropout(self):
        """
        Checks the existence of dropout layers in the models of the pipeline.

        Raises:
            ValueError: Raised when no dropout layers are found.
        """
        logger.debug("\n\n------------------ Testing Dropout --------------------")
        for model_key, pl in self.pipeline_map.items():
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

    def recognise(self, inp) -> dict[str, str]:
        """
        Function to perform OCR

        Args:
            inp: input

        Returns:
            dictionary of outputs
        """
        # Until the OCR data is available
        # TODO https://github.com/alan-turing-institute/ARC-SPICE/issues/14
        return {"outputs": inp["source_text"]}

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
        translator_outputs = self.translator(
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
        # get softmax of the logits to get token probabilities
        softmax_logits = softmax(translator_outputs[0]["raw_outputs"]["logits"], dim=-1)
        max_token_scores = torch.max(softmax_logits, dim=-1).values.squeeze(dim=0)
        # record the output and token probabilities
        confidence_metrics = [
            {
                "outputs": translator_output["translation_text"],
                "probs": max_token_scores,
            }
            for translator_output in translator_outputs
        ]
        # group metrics under a single dict key
        stacked_conf_metrics = self.stack_translator_sentence_metrics(
            confidence_metrics
        )
        # add full output to the output dict
        outputs: dict[str, Any] = {"full_output": full_translation}
        outputs.update(stacked_conf_metrics)
        # {full translation, sentence translations, logits, semantic embeddings}
        return outputs

    def classify_topic(self, text: str) -> dict[str, str]:
        """
        Runs the classification model

        Returns:
            Dictionary of classification outputs, namely the output scores.
        """
        forward = self.classifier(text, self.topic_labels)
        return {"scores": forward["scores"]}

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
        for var_index, var_score in enumerate(var_scores):
            cond_probs[var_index] = torch.prod(
                torch.pow(var_score, 1 / len(var_score)), dim=-1
            )
        # caclulate semantic density measure
        semantic_density = (1 / torch.sum(cond_probs)) * torch.sum(
            torch.mul(cond_probs, kernel_funcs)
        )

        return semantic_density.item(), sequence_length

    def translation_semantic_density(
        self, clean_output, var_output: dict
    ) -> dict[str, float | list[Any]]:
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
            }
        )

        return var_output

    def get_classification_confidence(
        self, var_output: dict, epsilon: float = 1e-15
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
            set_dropout(model=pl.model, dropout_flag=True)
            torch.nn.functional.dropout = dropout_on
            # do n runs of the inference
            for run_idx in range(self.n_variational_runs):
                var_output[model_key][run_idx] = self.func_map[model_key](
                    input_map[model_key]
                )
            # turn off dropout for this model
            set_dropout(model=pl.model, dropout_flag=False)
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

        return {"output_ids": output_ids, "logits": logits}
