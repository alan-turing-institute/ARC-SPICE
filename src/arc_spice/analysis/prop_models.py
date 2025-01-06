"""
Collecting the propagation models.
"""

from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression

from arc_spice.analysis.utils import brier_score, test_train_split_res


def eval_mult_prop(
    results_dict: dict[str, tuple[list[float], list[float]]],
) -> dict[str, float]:
    """Evaluate the multiplication prop version

    Args:
        results_dict: collated results dict, with structure:
                        {
                            'task': (uq vector, error vector)
                        }

    Returns:
        brier score for each step
    """
    mult_res = multiplication_prop(results_dict)
    out = {}
    for key, itm in results_dict.items():
        out[key] = brier_score(mult_res[key], itm[1])
    return out


def fit_uncertainty_model(
    uq_dict: dict[str, tuple[list[float], list[float]]],
) -> dict[str, LinearRegression]:
    """Recursively fit an uncertainty propagation model, outputting all three models.
    Model looks like:

        final_uq = E*(C*(A*OCR_uq + B) + D*translation_uq) + F*classification_uq

    where A, B, C, D, E and F will all be fit.

    NB: this is currently specific to our current pipeline of:

        recognition -> translation -> classification

    Args:
        uq_dict: dictionary of uncertainty quantifications for each stage of pipeline
                    to fit model with. Dict with structure:
                    {
                        'task': (uncertainties vector, error vector)
                    }

    Returns:
        fit_models_dict: dictionary of fitted models with structure:
                    {
                        'task': fitted linear model
                    }
    """
    # fit recognition step
    x1 = np.array(uq_dict["recognition"][0]).reshape(-1, 1)
    y1 = np.array(uq_dict["recognition"][1]).reshape(-1, 1)
    reg1 = LinearRegression(fit_intercept=True).fit(x1, y1)

    # fit translation step
    x2 = np.column_stack(
        (reg1.predict(x1), np.array(uq_dict["translation"][0]).reshape(-1, 1))
    )
    y2 = np.array(uq_dict["translation"][1]).reshape(-1, 1)
    reg2 = LinearRegression(fit_intercept=False).fit(x2, y2)

    # fit classification step
    x3 = np.column_stack(
        (reg2.predict(x2), np.array(uq_dict["classification"][0]).reshape(-1, 1))
    )
    y3 = np.array(uq_dict["classification"][1]).reshape(-1, 1)
    reg3 = LinearRegression(fit_intercept=False).fit(x3, y3)

    # return fitted models
    return {"recognition": reg1, "translation": reg2, "classification": reg3}


def multiplication_prop(results_dict, metric_map=None):
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
    if metric_map is None:
        metric_map = {
            "recognition": "mean_entropy",
            "translation": "weighted_semantic_density",
            "classification": "mean_predicted_entropy",
        }
    # split results and fit models
    vectors_dict = {
        "recognition": (
            1 - np.array(results_dict["recognition"][metric_map["recognition"]]),
            1 - np.array(results_dict["recognition"]["character_error_rate"]),
        ),
        "translation": (
            np.array(results_dict["translation"][metric_map["translation"]]),
            np.array(results_dict["translation"]["comet_score"]),
        ),
        "classification": (
            (
                1
                - np.array(results_dict["classification"][metric_map["classification"]])
            ).tolist(),
            (1 - np.array(results_dict["classification"]["hamming_loss"])).tolist(),
        ),
    }
    previous_vec = np.ones_like(np.array(vectors_dict["recognition"][0]))
    mult_res = {}
    for step_key in list(vectors_dict.keys()):
        previous_vec = previous_vec * np.array(vectors_dict[step_key][0])
        mult_res[step_key] = previous_vec
    return mult_res


def fitted_uq_model(results_dict, metric_map=None):
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
    if metric_map is None:
        metric_map = {
            "recognition": "mean_entropy",
            "translation": "weighted_semantic_density",
            "classification": "mean_predicted_entropy",
        }
    # split results and fit models
    vectors_dict = {
        "recognition": (
            1 - np.array(results_dict["recognition"][metric_map["recognition"]]),
            1 - np.array(results_dict["recognition"]["character_error_rate"]),
        ),
        "translation": (
            np.array(results_dict["translation"][metric_map["translation"]]),
            np.array(results_dict["translation"]["comet_score"]),
        ),
        "classification": (
            (
                1
                - np.array(results_dict["classification"][metric_map["classification"]])
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


def eval_lin_models(
    lin_uq_models: dict[str, LinearRegression],
    test_uq: dict[str, tuple[list[float], list[float]]],
) -> dict[Any, Any]:
    """Evaluate the uncertainty propagation models using the brier score

    Args:
        lin_uq_models: fitted linear uncertainty prop models
        test_uq: uncertainty quantifications on which to test

    Returns:
        dictionary of results
    """
    # recognition step
    x1 = np.array(test_uq["recognition"][0]).reshape(-1, 1)
    pred1 = lin_uq_models["recognition"].predict(x1)
    recog_brier = brier_score(pred1.reshape(1, -1), test_uq["recognition"][1])

    # translation step
    x2 = np.column_stack(
        (
            pred1,
            np.array(test_uq["translation"][0]).reshape(-1, 1),
        )
    )
    pred2 = lin_uq_models["translation"].predict(x2)
    trans_brier = brier_score(pred2.reshape(1, -1), test_uq["translation"][1])

    # classification step
    x3 = np.column_stack(
        (
            pred2,
            np.array(test_uq["classification"][0]).reshape(-1, 1),
        )
    )
    pred3 = lin_uq_models["classification"].predict(x3)
    class_brier = brier_score(pred3.reshape(1, -1), test_uq["classification"][1])

    return {
        "recognition": recog_brier,
        "translation": trans_brier,
        "classification": class_brier,
    }
