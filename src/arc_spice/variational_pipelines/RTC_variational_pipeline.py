import copy
import logging

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.functional import cosine_similarity, softmax
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TranslationPipeline,
    pipeline,
)

from arc_spice.variational_pipelines.dropout_utils import count_dropout, set_dropout

# From huggingface page with model:
#   - https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

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
        model_pars: dict[str : dict[str:str]],
        data_pars,
        n_variational_runs=5,
        translation_batch_size=8,
    ) -> None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        logging.info(f"Loading pipeline on device: {device}")

        self.OCR = pipeline(
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

        self.topic_labels = [
            class_names_dict["en"]
            for class_names_dict in data_pars["class_descriptors"]
        ]

        self._init_semantic_density()

        self.pipeline_map = {
            "recognition": self.OCR,
            "translation": self.translator,
            "classification": self.classifier,
        }

        self.func_map = {
            "recognition": self.recognise,
            "translation": self.translate,
            "classification": self.classify_topic,
        }
        self.naive_outputs = {
            "recognition": [
                "outputs",
            ],
            "translation": [
                "full_output",
                "outputs",
                "logits",
            ],
            "classification": [
                "scores",
            ],
        }
        self.n_variational_runs = n_variational_runs
        self.translation_batch_size = translation_batch_size

    def _init_semantic_density(self):
        self.nli_tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/deberta-large-mnli"
        )

        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-large-mnli"
        )

    @staticmethod
    def split_translate_inputs(text, split_key):
        split_rows = text.split(split_key)
        # for when string ends with with the delimiter
        if split_rows[-1] == "":
            split_rows = split_rows[:-1]
        recovered_splits = [split + split_key for split in split_rows]
        return recovered_splits

    def check_dropout(self):
        logger.debug("\n\n------------------ Testing Dropout --------------------")
        for model_key, pl in self.pipeline_map.items():
            # turn on dropout for this model
            set_dropout(model=pl.model, dropout_flag=True)
            logger.debug(f"Model key: {model_key}")
            dropout_count = count_dropout(pipe=pl, dropout_flag=True)
            logger.debug(
                f"{dropout_count} dropout layers found in correct configuration."
            )
            if dropout_count == 0:
                raise ValueError(f"No dropout layers found in {model_key}")
            set_dropout(model=pl.model, dropout_flag=False)
        logger.debug("-------------------------------------------------------\n\n")

    def recognise(self, inp):
        # Until the OCR data is available
        # TODO https://github.com/alan-turing-institute/ARC-SPICE/issues/14
        return {"outputs": inp}

    def translate(self, text):
        text_splits = self.split_translate_inputs(text, ".")

        translator_outputs = self.translator(
            text_splits,
            output_logits=True,
            return_dict_in_generate=True,
            batch_size=self.translation_batch_size,
        )
        sentence_translations = [
            translator_output["translation_text"]
            for translator_output in translator_outputs
        ]
        full_translation = ("").join(sentence_translations)

        confidence_metrics = [
            {
                "outputs": translator_output["translation_text"],
                "logits": translator_output["raw_outputs"]["logits"],
            }
            for translator_output in translator_outputs
        ]

        stacked_conf_metrics = self.stack_translator_sentence_metrics(
            confidence_metrics
        )
        outputs = {"full_output": full_translation}
        outputs.update(stacked_conf_metrics)
        # {full translation, sentence translations, logits, semantic embeddings}
        return outputs

    def classify_topic(self, text):
        forward = self.classifier(text, self.topic_labels)
        return {"scores": forward["scores"]}

    def stack_translator_sentence_metrics(self, all_sentence_metrics):
        stacked = {}
        for metric in self.naive_outputs["translation"][1:]:
            stacked[metric] = [
                sentence_metrics[metric] for sentence_metrics in all_sentence_metrics
            ]
        return stacked

    def stack_variational_outputs(self):
        new_var_dict = {}
        for step in self.var_output.keys():
            new_var_dict[step] = {}
            for metric in self.naive_outputs[step]:
                new_values = [None] * self.n_variational_runs
                for run in range(self.n_variational_runs):
                    new_values[run] = self.var_output[step][run][metric]
                new_var_dict[step][metric] = new_values

        self.var_output = new_var_dict

    def sentence_density(
        self,
        clean_sentence: str,
        var_sentences: list[str],
        var_scores: list[torch.Tensor],
    ) -> tuple[float, int]:
        # calc sequence lengths (for sentence weighting)
        sequence_length = len(
            self.nli_tokenizer.encode(
                clean_sentence, padding=True, return_tensors="pt"
            )[0]
        )

        kernel_funcs = torch.zeros(self.n_variational_runs)
        cond_probs = torch.zeros(self.n_variational_runs)

        for var_index, var_sentence in enumerate(var_sentences):
            nli_inp = clean_sentence + " [SEP] " + var_sentence
            encoded_nli = self.nli_tokenizer.encode(
                nli_inp, padding=True, return_tensors="pt"
            )
            nli_out = softmax(self.nli_model(encoded_nli)["logits"], dim=-1)[0]
            contradiction = nli_out[0]
            neutral = nli_out[1]

            kernel_funcs[var_index] = 1 - (contradiction + (0.5 * neutral))

        # TODO vectorize
        for var_index, var_sentence in enumerate(var_sentences):
            softmax_logits = softmax(var_scores[var_index], dim=-1)
            max_token_scores = torch.max(softmax_logits, dim=-1).values.squeeze(dim=0)
            cond_probs[var_index] = torch.pow(
                torch.prod(max_token_scores, dim=-1), 1 / len(max_token_scores)
            )

        semantic_density = (
            1 / (torch.sum(cond_probs)) * torch.sum(torch.mul(cond_probs, kernel_funcs))
        )
        return semantic_density.item(), sequence_length

    def translation_semantic_density(self):
        # from https://arxiv.org/pdf/2302.09664
        # github impl: https://github.com/lorenzkuhn/semantic_uncertainty

        # Broadly:

        # for each sentence in translation:
        #     use NLI model to determine semantic similarity to clean sentence
        #     (1 - (contradiction + 0.5 * neutral))

        # take mean, weighted by length of sentence

        # Need to loop over the sentences and calculate metrics comparing against clean
        # Average these (weighted by the sequence length?)
        # This is the overall confidence

        clean_out = self.clean_output["translation"]["outputs"]
        var_steps = self.var_output["translation"]
        n_sentences = len(clean_out)
        densities = [None] * n_sentences
        sequence_lengths = [None] * n_sentences
        for sentence_index, clean_sentence in enumerate(clean_out):
            var_sentences = [step[sentence_index] for step in var_steps["outputs"]]
            var_logits = [step[sentence_index] for step in var_steps["logits"]]
            density, seqlen = self.sentence_density(
                clean_sentence=clean_sentence,
                var_sentences=var_sentences,
                var_scores=var_logits,
            )
            densities[sentence_index] = density
            sequence_lengths[sentence_index] = seqlen

        total_len = torch.sum(torch.tensor(sequence_lengths))
        weighted_average = (
            torch.sum(torch.tensor(densities) * torch.tensor(sequence_lengths))
            / total_len
        )
        self.var_output["translation"].update(
            {
                "semantic_densities": densities,
                "weighted_semantic_density": weighted_average.item(),
            }
        )

    def get_classification_confidence(self, epsilon=1e-15):
        # From: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9761166
        stacked_probs = torch.stack(
            [torch.tensor(pred) for pred in self.var_output["classification"]["probs"]],
            dim=1,
        )

        means = torch.mean(stacked_probs, dim=-1)

        pred_entropies = -1 * (
            means * torch.log(means + epsilon)
            + (1 - means) * torch.log((1 - means) + epsilon)
        )

        all_entropies = -1 * (
            stacked_probs * torch.log(stacked_probs + epsilon)
            + (1 - stacked_probs) * torch.log((1 - stacked_probs) + epsilon)
        )

        mutual_info = pred_entropies - torch.mean(all_entropies, dim=-1)

        stds = torch.std(stacked_probs, dim=-1)
        variances = torch.var(stacked_probs, dim=-1)

        self.var_output["classification"].update(
            {
                "mean_scores": means,
                "std_scores": stds,
                "var_scores": variances,
                "predicted_entropy": pred_entropies,
                "mutual_information": mutual_info,
            }
        )

    def clean_inference(self, x):
        self.clean_output = {
            "recognition": {},
            "translation": {},
            "classification": {},
        }
        """Run the pipeline on an input x"""

        # UNTIL THE OCR DATA IS AVAILABLE
        self.clean_output["recognition"] = self.recognise(x)

        self.clean_output["translation"] = self.translate(
            self.clean_output["recognition"]["outputs"]
        )
        self.clean_output["classification"] = self.classify_topic(
            self.clean_output["translation"]["outputs"][0]
        )

    def variational_inference(self, x):
        self.clean_inference(x)
        self.var_output = {
            "recognition": [None] * self.n_variational_runs,
            "translation": [None] * self.n_variational_runs,
            "classification": [None] * self.n_variational_runs,
        }

        input_map = {
            "recognition": x,
            "translation": self.clean_output["recognition"]["outputs"],
            "classification": self.clean_output["translation"]["full_output"],
        }

        # for each model in pipeline
        for model_key, pl in self.pipeline_map.items():
            # turn on dropout for this model
            set_dropout(model=pl.model, dropout_flag=True)
            # do n runs of the inference
            for run_idx in range(self.n_variational_runs):
                self.var_output[model_key][run_idx] = self.func_map[model_key](
                    input_map[model_key]
                )
            # turn off dropout for this model
            set_dropout(model=pl.model, dropout_flag=False)

        self.stack_variational_outputs()
        self.translation_semantic_density()
        self.get_classification_confidence()

    def __call__(self, x):
        self.clean_inference(x)
        return self.clean_output


# Translation pipeline with additional functionality to save logits from fwd pass
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

        # logits are a tuple of length output_ids[-1]-1
        # each element is a tensor of shape (batch_size, vocab_size)
        logits = torch.stack(out["logits"], dim=1)

        return {"output_ids": output_ids, "logits": logits}
