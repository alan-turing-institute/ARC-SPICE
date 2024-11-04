import copy

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

from arc_spice.variational_pipelines.dropout_utils import set_dropout, test_dropout

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


class RTCVariationalPipeline:
    """
    variational version of the RTC pipeline
    """

    def __init__(
        self, model_pars: dict[str : dict[str:str]], data_pars, n_variational_runs=5
    ) -> None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        device = "cpu"
        print(f"Loading pipeline on device: {device}")

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
            "recognition": [
                "outputs",
                # "logits",
                # "entropy",
                # "normalised_entropy",
                # "probs",
                # "semantic_embedding",
            ],
            "translation": [
                "full_output",
                "outputs",
                "logits",
                "entropy",
                "normalised_entropy",
                "probs",
                "semantic_embedding",
            ],
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
        # for when string ends with with the delimiter
        if split_rows[-1] == "":
            split_rows = split_rows[:-1]
        recovered_splits = [split + split_key for split in split_rows]
        return recovered_splits

    def check_dropout(self):
        print("\n\n------------------ Testing Dropout --------------------")
        for model_key, pl in self.pipeline_map.items():
            # turn on dropout for this model
            set_dropout(model=pl.model, dropout_flag=True)
            print(f"Model key: {model_key}")
            test_dropout(pl, True)
            set_dropout(model=pl.model, dropout_flag=False)
        print("-------------------------------------------------------\n\n")

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
        # Until the OCR data is available
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

    def translation_semantic_density(self):
        # Broadly:
        # Need to loop over the sentences and calculate metrics comparing against clean
        #   Can we get these whilst we're running it? Probably will move this
        # Average these (weighted by the sequence length?)
        # This is the overall confidence

        clean_out = self.clean_output["translation"]["outputs"]
        var_step = self.var_output["translation"]
        n_sentences = len(clean_out)
        densities = [None] * n_sentences
        simalarities = [None] * n_sentences
        sequence_lengths = [None] * n_sentences
        for sentence_index, clean_sentence in enumerate(clean_out):
            sequence_lengths[sentence_index] = len(
                self.nli_tokenizer.encode(
                    clean_sentence, padding=True, return_tensors="pt"
                )[0]
            )
            kernel_funcs = torch.zeros(self.n_variational_runs)
            cond_probs = torch.zeros(self.n_variational_runs)
            sims = [None] * self.n_variational_runs
            run_sentences = [
                run_outputs[sentence_index] for run_outputs in var_step["outputs"]
            ]
            for run_index, run_out in enumerate(run_sentences):
                run_prob = var_step["probs"][run_index][0]
                nli_inp = clean_sentence + " [SEP] " + run_out
                encoded_nli = self.nli_tokenizer.encode(
                    nli_inp, padding=True, return_tensors="pt"
                )
                sims[run_index] = cosine_similarity(
                    self.clean_output["translation"]["semantic_embedding"][
                        sentence_index
                    ][0],
                    var_step["semantic_embedding"][run_index][sentence_index],
                )
                nli_out = softmax(self.nli_model(encoded_nli)["logits"], dim=-1)[0]
                kernel_funcs[run_index] = 1 - (nli_out[0] + (0.5 * nli_out[1]))
                cond_probs[run_index] = torch.pow(
                    torch.prod(run_prob, -1), 1 / len(run_prob)
                )
            semantic_density = (
                1
                / (torch.sum(cond_probs))
                * torch.sum(torch.mul(cond_probs, kernel_funcs))
            )
            densities[sentence_index] = semantic_density.item()
            simalarities[sentence_index] = [sim.item() for sim in sims]

        total_len = torch.sum(torch.tensor(sequence_lengths))
        weighted_average = (
            torch.sum(torch.tensor(densities) * torch.tensor(sequence_lengths))
            / total_len
        )
        self.var_output["translation"].update(
            {
                "semantic_densities": densities,
                "semantic_simalarities": simalarities,
                "sequence_lengths": sequence_lengths,
                "weighted_semantic_density": weighted_average.item(),
            }
        )

    def get_classification_confidence(self):
        all_preds = torch.stack(
            [torch.tensor(pred) for pred in self.var_output["classification"]["probs"]]
        )
        mean_scores = torch.mean(all_preds, dim=0)
        std_scores = torch.std(all_preds, dim=0)
        self.var_output["classification"].update(
            {
                "mean_scores": mean_scores,
                "std_scores": std_scores,
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
        self.clean_output["classification"] = self.classify(
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
