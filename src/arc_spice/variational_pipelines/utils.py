import logging
from functools import partial
from typing import Any

import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Pipeline

logger = logging.Logger("RTC_variational_pipeline")


class DummyPipeline:
    """
    For initialising a base pipeline which needs to be overwritten by a subclass
    """

    def __init__(self, model_name):
        """
        Gives the dummy pipeline the required attributes for the method definitions

        Args:
            model_name: name of the pipeline that is being given a dummy
        """
        self.model = model_name

    def __call__(self, *args, **kwargs):
        """
        Needs to be defined in subclass

        Raises:
            NotImplementedError: when called to prevent base class being used
        """
        error_msg = (
            f"{self.model} cannot be called directly and needs to be"
            " defined in a subclass."
        )
        raise NotImplementedError(error_msg)


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


class RTCPipelineBase:
    def __init__(self, n_variational_runs=5, translation_batch_size=8):
        # device for inference
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        debug_msg_device = f"Loading pipeline on device: {self.device}"
        logging.info(debug_msg_device)
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

        self.ocr = DummyPipeline("ocr")
        self.translator = DummyPipeline("translator")
        self.classifier = DummyPipeline("classifier")

        # map pipeline names to their pipeline counterparts

        self.topic_labels = None  # This should be defined in subclass if needed
        self._init_pipeline_map()

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
            all_sentence_metrics=confidence_metrics
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
