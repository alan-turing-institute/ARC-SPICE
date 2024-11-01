from torcheval.metrics.functional import bleu_score


def get_bleu_score(target, translation):
    return bleu_score(target, translation, n_gram=4).item()
