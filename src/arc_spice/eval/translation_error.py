from comet import download_model, load_from_checkpoint
from torcheval.metrics.functional import bleu_score


def get_bleu_score(target, translation):
    return bleu_score(target, translation, n_gram=4).item()


def get_comet_model(model_path="Unbabel/wmt22-comet-da"):
    # Load the model checkpoint:
    comet_model_pth = download_model(model=model_path)
    comet_model = load_from_checkpoint(comet_model_pth)
    return comet_model
