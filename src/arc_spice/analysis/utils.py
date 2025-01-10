import numpy as np
import torch
from sklearn.model_selection import train_test_split

from arc_spice.eval.classification_error import clean_entropy, max_scores
from arc_spice.eval.translation_error import length_normalised_metric

steps = ["recognition", "translation", "classification"]


def recognition_analysis(all_results):
    return create_results_dict(*recognition_vectors(all_results))


def translation_analysis(all_results):
    return create_results_dict(*translation_vectors(all_results))


def classification_analysis(all_results):
    return create_results_dict(*classification_vectors(all_results))


analysis_func_map = {
    "ocr": recognition_analysis,
    "translator": translation_analysis,
    "classifier": classification_analysis,
}


def exp_analysis(results_dict: dict, analysis_keys: list):
    return {key: analysis_func_map[key](results_dict) for key in analysis_keys}


def mean_square_error(predicted: list, error: list):
    # do brier score calculation
    return torch.mean(
        torch.pow((torch.tensor(predicted) - torch.tensor(error)), 2)
    ).item()


def get_vectors(
    all_results: list[dict], step_key: str, target_celex_ids: None | np.ndarray = None
):
    # instantiate lists so we can add elements in custom order if needed
    vector_lengths = (
        len(all_results) if target_celex_ids is None else len(target_celex_ids)
    )
    vector_dict = {
        key: [None] * vector_lengths
        for key in next(iter(all_results[0].values()))[step_key]
    }
    vector_dict["celex_id"] = [None] * vector_lengths

    if target_celex_ids is not None:
        # if we have target ids we need to ensure order is maintained across all lists
        target_celex_ids = target_celex_ids.tolist()
        for row_dict in all_results:
            row_values = next(iter(row_dict.values()))
            row_celex_id = next(iter(row_dict.keys()))
            try:
                target_index = target_celex_ids.index(row_celex_id)
            except ValueError:
                # skip row if we don't have an index from target ids
                continue
            # put all values into the target index position
            for key in vector_dict:
                if key == "celex_id":
                    # row_values doesn't contain celex_id
                    vector_dict[key][target_index] = row_celex_id
                else:
                    vector_dict[key][target_index] = row_values[step_key][key]
    else:
        # otherwise just fill according to non-split order
        for row_index, row_dict in enumerate(all_results):
            row_values = next(iter(row_dict.values()))
            row_celex_id = next(iter(row_dict.keys()))
            for key in vector_dict:
                if key == "celex_id":
                    # row_values doesn't contain celex_id
                    vector_dict["celex_id"][row_index] = row_celex_id
                else:
                    vector_dict[key][row_index] = row_values[step_key][key]

    if step_key == "recognition":
        vector_dict["mean_confidence"] = (
            1 - np.array(vector_dict["mean_entropy"])
        ).tolist()
        vector_dict["character_accuracy_rate"] = (
            1 - np.array(vector_dict["character_error_rate"])
        ).tolist()

    elif step_key == "translation":
        # additional measures
        vector_dict["len_norm_cond_prob"] = length_normalised_metric(
            vector_dict["sequence_lengths"],
            vector_dict["clean_conditional_probability"],
        )
        vector_dict["len_norm_entropy"] = length_normalised_metric(
            vector_dict["sequence_lengths"],
            vector_dict["mean_entropy"],
        )
        vector_dict["len_norm_confidence"] = 1 - np.array(
            vector_dict["len_norm_entropy"]
        )

    elif step_key == "classification":
        # additional measures
        vector_dict["clean_entropy"] = clean_entropy(
            vector_dict["clean_scores"],
        ).tolist()
        # additional measures
        vector_dict["clean_confidence"] = (
            1 - clean_entropy(vector_dict["clean_scores"])
        ).tolist()
        vector_dict["mean_predicted_confidence"] = (
            1 - np.array(vector_dict["mean_predicted_entropy"])
        ).tolist()
        vector_dict["hamming_accuracy"] = (
            1 - np.array(vector_dict["hamming_loss"])
        ).tolist()
        vector_dict["clean_predicted_scores"] = max_scores(vector_dict["clean_scores"])
        vector_dict["mean_predicted_scores"] = max_scores(vector_dict["mean_scores"])

    return vector_dict


def test_train_split_res(
    results_dict: dict[str, tuple[list[float], list[float]]], seed: int = 37
) -> tuple[
    dict[str, tuple[list[float], list[float]]],
    dict[str, tuple[list[float], list[float]]],
]:
    """Split a collection of extracted results dictionaries into a test/train split.

    Args:
        results_dict: collected results vector dictionary with structure
                        {
                            'task': (uq vectors, error vectors)
                        }
    """
    train_res = {}
    test_res = {}
    for key, itm in results_dict.items():
        split = train_test_split(np.column_stack(itm), random_state=seed)
        train_res[key] = (split[0][:, 0], split[0][:, 1])
        test_res[key] = (split[1][:, 0], split[1][:, 1])
    return train_res, test_res


def exp_vectors(results_dict: list[dict], analysis_keys: list, **kwargs):
    return {
        key: get_vectors(all_results=results_dict, step_key=key, **kwargs)
        for key in analysis_keys
    }


def recognition_vectors(all_results: list[dict]):
    # brier score params:   1 - entropy, 1 - character error rate
    confidence_vector = [
        (1 - next(iter(sample_dict.values()))["recognition"]["mean_entropy"])
        for sample_dict in all_results
    ]
    accuracy_vector = [
        (1 - next(iter(sample_dict.values()))["recognition"]["character_error_rate"])
        for sample_dict in all_results
    ]
    return confidence_vector, accuracy_vector


def translation_vectors(all_results: list[dict]):
    # brier score params:   semantic_density, comet score
    confidence_vector = [
        next(iter(sample_dict.values()))["translation"]["weighted_semantic_density"]
        for sample_dict in all_results
    ]
    accuracy_vector = [
        next(iter(sample_dict.values()))["translation"]["comet_score"]
        for sample_dict in all_results
    ]
    return confidence_vector, accuracy_vector


def classification_vectors(all_results: list[dict]):
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

    return confidence_vector, accuracy_vector


def create_results_dict(confidence_vector, accuracy_vector):
    return {
        "mean_square_error": mean_square_error(
            predicted=confidence_vector, error=accuracy_vector
        ),
        "mean_accuracy": sum(accuracy_vector) / len(accuracy_vector),
    }


def collect_pipeline_dict(
    results_dict: list[dict],
) -> dict[str, tuple[list[float], list[float]]]:
    """
    Given a loaded results dict for the entire pipeline collect all three sets of
    results into a dictionary.
    """

    return {
        "recognition": recognition_vectors(results_dict),
        "translation": translation_vectors(results_dict),
        "classification": classification_vectors(results_dict),
    }
