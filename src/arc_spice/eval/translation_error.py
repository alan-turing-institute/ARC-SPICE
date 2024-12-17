import torch
from comet import download_model, load_from_checkpoint
from torcheval.metrics.functional import bleu_score


def get_bleu_score(target, translation):
    return bleu_score(target, translation, n_gram=4).item()


def get_comet_model(model_path="Unbabel/wmt22-comet-da"):
    # Load the model checkpoint:
    comet_model_pth = download_model(model=model_path)
    return load_from_checkpoint(comet_model_pth)


def conditional_probability(prob_scores: torch.Tensor):
    return torch.prod(torch.pow(prob_scores, 1 / len(prob_scores)), dim=-1)


def length_normalised_metric(sequence_lengths, metric):
    vals = torch.zeros(len(sequence_lengths))
    for index, (row_sequence_lengths, row_metric) in enumerate(
        zip(sequence_lengths, metric, strict=True)
    ):
        seq_len = torch.tensor(row_sequence_lengths)
        probs = torch.tensor(row_metric)

        length_normalised_metric = torch.sum(seq_len * probs) / torch.sum(seq_len)
        vals[index] = length_normalised_metric.item()

    return vals
