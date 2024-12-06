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
