import torch


def recognition_analysis(all_results: list[dict]):
    # brier score params:   1 - entropy, 1 - character error rate
    confidence_vector = [
        (1 - next(iter(sample_dict.values()))["recognition"]["mean_entropy"])
        for sample_dict in all_results
    ]
    accuracy_vector = [
        (1 - next(iter(sample_dict.values()))["recognition"]["character_error_rate"])
        for sample_dict in all_results
    ]
    return create_results_dict(confidence_vector, accuracy_vector)


def translation_analysis(all_results: list[dict]):
    # brier score params:   semantic_density, comet score
    confidence_vector = [
        next(iter(sample_dict.values()))["translation"]["weighted_semantic_density"]
        for sample_dict in all_results
    ]
    accuracy_vector = [
        next(iter(sample_dict.values()))["translation"]["comet_score"]
        for sample_dict in all_results
    ]
    return create_results_dict(confidence_vector, accuracy_vector)


def classification_analysis(all_results: list[dict]):
    # brier score params:   1 - entropy, hamming accuracy/zero-one-accuracy
    confidence_vector = [
        (
            1
            - next(iter(sample_dict.values()))["classification"][
                "mean_predicted_entropy"
            ]
        )
        for sample_dict in all_results
    ]
    accuracy_vector = [
        (1 - next(iter(sample_dict.values()))["classification"]["hamming_loss"])
        for sample_dict in all_results
    ]

    return create_results_dict(confidence_vector, accuracy_vector)


def create_results_dict(confidence_vector, accuracy_vector):
    return {
        "brier_score": brier_score(predicted=confidence_vector, error=accuracy_vector),
        "mean_accuracy": sum(accuracy_vector) / len(accuracy_vector),
    }


def brier_score(predicted: list, error: list):
    # do brier score calculation
    return torch.mean(
        torch.pow((torch.tensor(predicted) - torch.tensor(error)), 2)
    ).item()


analysis_func_map = {
    "ocr": recognition_analysis,
    "translator": translation_analysis,
    "classifier": classification_analysis,
}


def exp_analysis(results_dict: list[dict], analysis_keys: list):
    return {key: analysis_func_map[key](results_dict) for key in analysis_keys}
