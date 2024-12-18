import numpy as np

from arc_spice.eval.analysis_utils import test_train_split_res
from arc_spice.eval.prop_models import fit_uncertainty_model

steps = ["recognition", "translation", "classification"]


def multiplication_prop(results_dict):
    """Compute the naïve approach of multiplying together confidences as though they
    represent independent probabilities of success.

    Args:
        results_dict: collated results dict, with structure:
                        {
                            'task': (uq vector, error vector)
                        }

    Returns:
        multiplied results dict, with same structure but updated uq vectors
    """
    # split results and fit models
    vectors_dict = {
        "recognition": (
            1 - np.array(results_dict["recognition"]["mean_entropy"]),
            1 - np.array(results_dict["recognition"]["character_error_rate"]),
        ),
        "translation": (
            np.array(results_dict["translation"]["weighted_semantic_density"]),
            np.array(results_dict["translation"]["comet_score"]),
        ),
        "classification": (
            1 - np.array(results_dict["classification"]["mean_predicted_entropy"]),
            1 - np.array(results_dict["classification"]["hamming_loss"]),
        ),
    }
    previous_vec = np.ones_like(np.array(vectors_dict["recognition"][0]))
    mult_res = {}
    for step_key in steps:
        previous_vec = previous_vec * np.array(vectors_dict[step_key][0])
        mult_res[step_key] = previous_vec
    return mult_res


def fitted_uq_model(results_dict):
    """Fit the uq models using the fit uncertainty models method on a test/train split,
    then populate the data with the test split

    Args:
        results_dict: collated results dict, with structure,
                        {
                            'task': (uq vector, error vector)
                        }

    Returns:
        test results split with uq propagation from fitted model
    """
    # split results and fit models
    vectors_dict = {
        "recognition": (
            (1 - np.array(results_dict["recognition"]["mean_entropy"])).tolist(),
            (
                1 - np.array(results_dict["recognition"]["character_error_rate"])
            ).tolist(),
        ),
        "translation": (
            results_dict["translation"]["weighted_semantic_density"],
            results_dict["translation"]["comet_score"],
        ),
        "classification": (
            (
                1 - np.array(results_dict["classification"]["mean_predicted_entropy"])
            ).tolist(),
            (1 - np.array(results_dict["classification"]["hamming_loss"])).tolist(),
        ),
        "celex_ids": (
            results_dict["classification"]["celex_id"],
            results_dict["classification"]["celex_id"],
        ),
    }
    train_res, test_res = test_train_split_res(vectors_dict)
    uq_models = fit_uncertainty_model(train_res)

    # generated predicted data
    recog_pred = uq_models["recognition"].predict(
        np.array(test_res["recognition"][0]).reshape(-1, 1)
    )
    trans_pred = uq_models["translation"].predict(
        np.column_stack(
            (
                recog_pred,
                np.array(test_res["translation"][0]).reshape(-1, 1),
            )
        )
    )
    class_pred = uq_models["classification"].predict(
        np.column_stack(
            (
                trans_pred,
                np.array(test_res["classification"][0]).reshape(-1, 1),
            )
        )
    )

    # return collated output
    return (
        {
            "recognition": (
                recog_pred.reshape(1, -1).squeeze(),
                test_res["recognition"][1],
            ),
            "translation": (
                trans_pred.reshape(1, -1).squeeze(),
                test_res["translation"][1],
            ),
            "classification": (
                class_pred.reshape(1, -1).squeeze(),
                test_res["classification"][1],
            ),
        },
        {
            "train_ids": train_res["celex_ids"][0],
            "test_ids": test_res["celex_ids"][0],
        },
    )
