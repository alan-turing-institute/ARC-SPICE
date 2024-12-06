import torch


def recognition_analysis(_: dict):
    # brier score params:   1 - entropy, character error rate
    raise NotImplementedError


def translation_analysis(translation_dict: dict):
    # brier score params:   semantic_density, comet score
    confidence_vector = [
        sample_dict["weighted_semantic_density"] for sample_dict in translation_dict
    ]
    accuracy_vector = [sample_dict["comet_score"] for sample_dict in translation_dict]
    return brier_score(confidence_vector, accuracy_vector)


def classification_analysis(classification_dict: dict):
    # brier score params:   1 - entropy, hamming accuracy/zero-one-accuracy
    confidence_vector = [
        (1 - sample_dict["mean_entropy"]) for sample_dict in classification_dict
    ]
    accuracy_vector = [
        (1 - sample_dict["hamming_loss"]) for sample_dict in classification_dict
    ]
    return brier_score(predicted=confidence_vector, error=accuracy_vector)


def brier_score(predicted: list, error: list):
    # do brier score calculation
    return torch.mean(torch.pow((torch.tensor(predicted) - torch.tensor(error)), 2))


analysis_func_map = {
    "ocr": recognition_analysis,
    "translation": translation_analysis,
    "classification": classification_analysis,
}


def brier_score_analysis(results_dict: dict, analysis_keys: list):
    return {key: analysis_func_map[key](results_dict[key]) for key in analysis_keys}
