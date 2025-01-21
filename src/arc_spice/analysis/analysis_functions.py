import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import RBF

from arc_spice.analysis.prop_models import (
    eval_lin_models,
    eval_mult_prop,
    fit_uncertainty_model,
    fitted_gp_model,
    fitted_lin_model,
    multiplication_prop,
)
from arc_spice.analysis.utils import (
    collect_pipeline_dict,
    exp_analysis,
    exp_vectors,
    test_train_split_res,
)
from arc_spice.utils import open_json_path


def propagation_analysis(experiment_path: str):
    """Run analysis of a given pipeline experiment using the different propagation
    models

    Args:
        experiment_path: path experiment directory
    """
    model_keys = ["ocr", "translator", "classifier"]

    # collect and collate results
    pipeline_results = open_json_path(f"{experiment_path}/full_pipeline.json")
    pipe_results = collect_pipeline_dict(pipeline_results)

    # no model results, rename keys
    no_mod_res = exp_analysis(pipe_results, model_keys)
    no_mod_res["recognition"] = no_mod_res.pop("ocr")
    no_mod_res["translation"] = no_mod_res.pop("translator")
    no_mod_res["classification"] = no_mod_res.pop("classifier")

    # multplication model resuls
    multi_mod_res = eval_mult_prop(pipe_results)

    # fitted model results
    train_res, test_res = test_train_split_res(pipe_results)
    fitted_uq_models = fit_uncertainty_model(train_res)
    fit_mod_res = eval_lin_models(fitted_uq_models, test_res)

    # collate results
    out_res = {}
    for key, itm in fit_mod_res.items():
        out_res[key] = {
            "no_model": no_mod_res[key],
            "mult_model": multi_mod_res[key],
            "fitted_model": itm,
        }
    return out_res


def single_model_analysis(
    experiment_path: str,
):
    """
    Run analysis on a given experiment with provided experiment path

    Args:
        experiment_path: path to experiment directory containing:
        - full_pipeline.json
        - ocr.json
        - translator.json
        - classifier.json
    """
    model_keys = ["ocr", "translator", "classifier"]

    # pipeline analysis
    pipeline_results = open_json_path(f"{experiment_path}/full_pipeline.json")
    pipe_results = exp_analysis(pipeline_results, model_keys)

    # individual component analysis
    ind_results = {}
    for model in model_keys:
        model_results = open_json_path(f"{experiment_path}/{model}.json")
        ind_results.update(exp_analysis(model_results, [model]))

    # create single dict
    return {
        "individual_components": ind_results,
        "pipeline": pipe_results,
    }


def k_fold_error_propagation_analysis(
    experiment_path: str,
    **kwargs,
):
    """
    ------------- WIP ----------------

    Run error propagation k times for a given experiment with provided experiment path

    Args:
        experiment_path: path to experiment directory containing:
        - full_pipeline.json
        - ocr.json
        - translator.json
        - classifier.json
    """
    step_keys = ["recognition", "translation", "classification"]

    pipeline_results = open_json_path(f"{experiment_path}/full_pipeline.json")

    vectors_w_lin, lin_celex_ids = fitted_lin_model(
        results_dict=exp_vectors(pipeline_results, step_keys), **kwargs
    )
    kwargs.pop("splits")
    pipeline_vectors = exp_vectors(
        pipeline_results, step_keys, target_celex_ids=lin_celex_ids["test_ids"]
    )
    multiplication_vectors = multiplication_prop(pipeline_vectors, **kwargs)
    for key in pipeline_vectors:
        pipeline_vectors[key]["linear_confidence"] = vectors_w_lin[key][0]
        pipeline_vectors[key]["multiplication_confidence"] = multiplication_vectors[key]

    classifier_vectors = exp_vectors(
        open_json_path(f"{experiment_path}/classifier.json"),
        ["classification"],
        target_celex_ids=lin_celex_ids["test_ids"],
    )
    translation_vectors = exp_vectors(
        open_json_path(f"{experiment_path}/translator.json"),
        ["translation"],
        target_celex_ids=lin_celex_ids["test_ids"],
    )
    return pipeline_vectors, translation_vectors, classifier_vectors


def error_propagation_analysis(experiment_path: str, **kwargs):
    """
    Run analysis on a given experiment with provided experiment path

    Args:
        experiment_path: path to experiment directory containing:
        - full_pipeline.json
        - ocr.json
        - translator.json
        - classifier.json
    """
    step_keys = ["recognition", "translation", "classification"]

    pipeline_results = open_json_path(f"{experiment_path}/full_pipeline.json")

    vectors_w_lin, lin_celex_ids = fitted_lin_model(
        results_dict=exp_vectors(pipeline_results, step_keys), **kwargs
    )
    vectors_w_gp, gp_celex_ids = fitted_gp_model(
        results_dict=exp_vectors(pipeline_results, step_keys), **kwargs
    )

    kwargs["kernel"] = RBF(length_scale_bounds=[1e-175, 100])
    vectors_w_gp_rbf, gp_rbf_celex_ids = fitted_gp_model(
        results_dict=exp_vectors(pipeline_results, step_keys),
        **kwargs,
    )
    kwargs.pop("kernel")

    for lin_id, gp_id, gp_rbf_id in zip(
        lin_celex_ids, gp_celex_ids, gp_rbf_celex_ids, strict=True
    ):
        assert lin_id == gp_id == gp_rbf_id

    pipeline_vectors = exp_vectors(
        pipeline_results, step_keys, target_celex_ids=lin_celex_ids["test_ids"]
    )
    multiplication_vectors = multiplication_prop(pipeline_vectors, **kwargs)
    for key in pipeline_vectors:
        pipeline_vectors[key]["linear_confidence"] = vectors_w_lin[key][0]
        pipeline_vectors[key]["gaussian_lin_confidence"] = vectors_w_gp[key][0]
        pipeline_vectors[key]["gaussian_rbf_confidence"] = vectors_w_gp_rbf[key][0]
        pipeline_vectors[key]["multiplication_confidence"] = multiplication_vectors[key]

    classifier_vectors = exp_vectors(
        open_json_path(f"{experiment_path}/classifier.json"),
        ["classification"],
        target_celex_ids=lin_celex_ids["test_ids"],
    )
    translation_vectors = exp_vectors(
        open_json_path(f"{experiment_path}/translator.json"),
        ["translation"],
        target_celex_ids=lin_celex_ids["test_ids"],
    )
    return pipeline_vectors, translation_vectors, classifier_vectors


def plot_error_propagation(all_propagation_results, save_directory, combination_keys):
    os.makedirs(
        f"{save_directory}/figures/propagation_mean_square_errors/", exist_ok=True
    )
    for combination_key in combination_keys:
        propagation_results = all_propagation_results[combination_key]
        no_model = (
            np.array(
                [
                    np.mean(next(iter(step.values())))
                    for step in propagation_results.values()
                ]
            ),
            np.array(
                [
                    np.std(next(iter(step.values())))
                    for step in propagation_results.values()
                ]
            ),
        )

        mult_model = (
            np.array(
                [
                    np.mean(step["multiplication_confidence"])
                    for step in propagation_results.values()
                ]
            ),
            np.array(
                [
                    np.std(step["multiplication_confidence"])
                    for step in propagation_results.values()
                ]
            ),
        )
        linear_model = (
            np.array(
                [
                    np.mean(step["linear_confidence"])
                    for step in propagation_results.values()
                ]
            ),
            np.array(
                [
                    np.std(step["linear_confidence"])
                    for step in propagation_results.values()
                ]
            ),
        )

        plt.plot([0, 1, 2], no_model[0], label="No Model")
        plt.fill_between(
            [0, 1, 2], no_model[0] - no_model[1], no_model[0] + no_model[1], alpha=0.2
        )
        plt.plot([0, 1, 2], mult_model[0], label="Multiplication Model")
        plt.fill_between(
            [0, 1, 2],
            mult_model[0] - mult_model[1],
            mult_model[0] + mult_model[1],
            alpha=0.2,
        )
        plt.plot([0, 1, 2], linear_model[0], label="Linear Model")
        plt.fill_between(
            [0, 1, 2],
            linear_model[0] - linear_model[1],
            linear_model[0] + linear_model[1],
            alpha=0.2,
        )
        plt.legend()
        plt.xticks([0, 1, 2], labels=["Recognition", "Translation", "Classification"])
        plt.xlabel("Step")
        plt.ylabel("RMSE")
        plt.savefig(
            f"{save_directory}/figures/propagation_mean_square_errors/"
            f"{combination_key}.pdf"
        )
        plt.close()


def plot_vectors(
    save_directory, pipeline_vectors, translator_vectors, classifier_vectors
):
    alph = 0.5
    n_bins = 75

    # recognition
    plt.hist(
        pipeline_vectors["recognition"]["character_accuracy_rate"],
        alpha=alph,
        label="CER",
        bins=n_bins,
    )
    plt.hist(
        pipeline_vectors["recognition"]["mean_confidence"],
        alpha=alph,
        label="Mean Entropy",
        bins=n_bins,
    )
    plt.hist(
        pipeline_vectors["recognition"]["linear_confidence"],
        alpha=alph,
        label="Linear fit",
        bins=n_bins,
    )
    # plt.hist(
    #     pipeline_vectors["recognition"]["gaussian_lin_confidence"],
    #     alpha=alph,
    #     label="Gaussian fit",
    #     bins=n_bins,
    # )
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.legend()
    plt.xlim(0, 1)
    plt.savefig(f"{save_directory}/figures/recognition_confidence_histogram.pdf")
    plt.close()

    # Translation
    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(8, 12))

    counts_list = []
    ax1.set_title("Pipeline", fontsize=20)
    ax2.set_title("Single Component", fontsize=20)

    counts, _, _ = ax1.hist(
        pipeline_vectors["translation"]["comet_score"],
        alpha=alph,
        label="Comet score",
        color="C0",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)
    counts, _, _ = ax1.hist(
        pipeline_vectors["translation"]["weighted_semantic_density"],
        alpha=alph,
        label="Semantic Density",
        color="C1",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)
    # counts, _, _ = ax1.hist(
    #     pipeline_vectors["translation"]["len_norm_cond_prob"],
    #     alpha=alph,
    #     label="LNCP",
    #     color="C2",
    #     bins=np.linspace(0, 1, n_bins),
    # )
    # counts_list.append(counts)
    # counts, _, _ = ax1.hist(
    #     1 - pipeline_vectors["translation"]["len_norm_entropy"],
    #     alpha=alph,
    #     label="LNE",
    #     color="C3",
    #     bins=np.linspace(0, 1, n_bins),
    # )
    # counts_list.append(counts)

    counts, _, _ = ax1.hist(
        pipeline_vectors["translation"]["multiplication_confidence"],
        alpha=alph,
        label="Multiplication",
        color="C2",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)
    counts, _, _ = ax1.hist(
        pipeline_vectors["translation"]["linear_confidence"],
        alpha=alph,
        label="Linear fit",
        color="C3",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)
    # counts, _, _ = ax1.hist(
    #     pipeline_vectors["translation"]["gaussian_lin_confidence"],
    #     alpha=alph,
    #     label="Gaussian fit",
    #     color="C6",
    #     bins=np.linspace(0, 1, n_bins),
    # )
    # counts_list.append(counts)

    counts, _, _ = ax2.hist(
        translator_vectors["translation"]["weighted_semantic_density"],
        alpha=alph,
        color="C1",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)
    # counts, _, _ = ax2.hist(
    #     translator_vectors["translation"]["len_norm_cond_prob"],
    #     alpha=alph,
    #     label="LNCP",
    #     color="C2",
    #     bins=np.linspace(0, 1, n_bins),
    # )
    # counts_list.append(counts)
    # counts, _, _ = ax2.hist(
    #     1 - translator_vectors["translation"]["len_norm_entropy"],
    #     alpha=alph,
    #     label="LNE",
    #     color="C3",
    #     bins=np.linspace(0, 1, n_bins),
    # )
    # counts_list.append(counts)
    counts, _, _ = ax2.hist(
        translator_vectors["translation"]["comet_score"],
        alpha=alph,
        color="C0",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)

    ax1.set_xlabel("Score", fontsize=18)
    ax1.set_ylabel("Count", fontsize=18)
    ax2.set_ylabel("Count", fontsize=18)
    ax1.legend(title="Metric")
    max_val = max([max(count) for count in counts_list]) + 10
    ax1.set_ylim(0, max_val)
    ax2.set_ylim(0, max_val)

    ax1.set_xlim(0, 1)
    ax2.set_xlim(0, 1)
    plt.savefig(f"{save_directory}/figures/translation_confidence_histogram.pdf")
    plt.close()

    fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(8, 12))

    counts_list = []
    ax1.set_title("Pipeline", fontsize=20)
    ax2.set_title("Single Component", fontsize=20)

    #  Plot classification results
    counts, _, _ = ax1.hist(
        pipeline_vectors["classification"]["hamming_accuracy"],
        alpha=alph,
        color="C0",
        label="Hamming Accuracy",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)
    counts, _, _ = ax1.hist(
        pipeline_vectors["classification"]["mean_predicted_confidence"],
        alpha=alph,
        color="C1",
        label="Confidence",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)
    # counts, _, _ = ax1.hist(
    #     pipeline_vectors["classification"]["clean_confidence"],
    #     alpha=alph,
    #     color="C2",
    #     label="Clean Confidence",
    #     bins=np.linspace(0, 1, n_bins),
    # )
    # counts_list.append(counts)
    counts, _, _ = ax1.hist(
        pipeline_vectors["classification"]["multiplication_confidence"],
        alpha=alph,
        color="C2",
        label="Multiplication",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)
    counts, _, _ = ax1.hist(
        pipeline_vectors["classification"]["linear_confidence"],
        alpha=alph,
        color="C3",
        label="Linear Fit",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)
    # counts, _, _ = ax1.hist(
    #     pipeline_vectors["translation"]["gaussian_lin_confidence"],
    #     alpha=alph,
    #     label="Gaussian fit",
    #     color="C6",
    #     bins=np.linspace(0, 1, n_bins),
    # )
    # counts_list.append(counts)
    counts, _, _ = ax2.hist(
        classifier_vectors["classification"]["hamming_accuracy"],
        alpha=alph,
        color="C0",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)
    counts, _, _ = ax2.hist(
        classifier_vectors["classification"]["mean_predicted_confidence"],
        alpha=alph,
        color="C1",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)
    # counts, _, _ = ax2.hist(
    #     classifier_vectors["classification"]["clean_confidence"],
    #     alpha=alph,
    #     color="C2",
    #     bins=np.linspace(0, 1, n_bins),
    # )
    # counts_list.append(counts)
    ax1.legend(title="Metric")

    ax1.set_xlabel("Score", fontsize=18)
    ax1.set_ylabel("Count", fontsize=18)
    ax2.set_ylabel("Count", fontsize=18)
    ax1.legend(title="Metric")
    max_val = max([max(count) for count in counts_list]) + 10
    ax1.set_ylim(0, max_val)
    ax2.set_ylim(0, max_val)

    ax1.set_xlim(0, 1)
    ax2.set_xlim(0, 1)
    plt.savefig(f"{save_directory}/figures/classification_confidence_histogram.pdf")
    plt.close()
