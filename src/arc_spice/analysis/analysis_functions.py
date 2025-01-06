import matplotlib.pyplot as plt
import numpy as np

from arc_spice.analysis.utils import fitted_lin_model, multiplication_prop
from arc_spice.eval.analysis_utils import exp_analysis, exp_vectors
from arc_spice.utils import open_json_path


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

    vectors_w_lin, celex_ids = fitted_lin_model(
        results_dict=exp_vectors(pipeline_results, step_keys), **kwargs
    )
    pipeline_vectors = exp_vectors(
        pipeline_results, step_keys, target_celex_ids=celex_ids["test_ids"]
    )
    multiplication_vectors = multiplication_prop(pipeline_vectors, **kwargs)
    for key in pipeline_vectors:
        pipeline_vectors[key]["linear_confidence"] = vectors_w_lin[key][0]
        pipeline_vectors[key]["multiplication_confidence"] = multiplication_vectors[key]

    classifier_vectors = exp_vectors(
        open_json_path(f"{experiment_path}/classifier.json"),
        ["classification"],
        target_celex_ids=celex_ids["test_ids"],
    )
    translation_vectors = exp_vectors(
        open_json_path(f"{experiment_path}/translator.json"),
        ["translation"],
        target_celex_ids=celex_ids["test_ids"],
    )
    return pipeline_vectors, translation_vectors, classifier_vectors


def plot_vectors(
    save_directory, pipeline_vectors, translator_vectors, classifier_vectors
):
    alph = 0.5
    n_bins = 75

    # recognition
    plt.title("Recognition")
    plt.hist(
        pipeline_vectors["recognition"]["mean_confidence"],
        alpha=alph,
        label="Mean Entropy",
        bins=n_bins,
    )
    plt.hist(
        pipeline_vectors["recognition"]["character_accuracy_rate"],
        alpha=alph,
        label="CER",
        bins=n_bins,
    )
    plt.hist(
        pipeline_vectors["recognition"]["linear_confidence"],
        alpha=alph,
        label="linear fit",
        bins=n_bins,
    )
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.legend()
    plt.xlim(0, 1)
    plt.savefig(f"{save_directory}/figures/recognition_confidence_histogram.pdf")

    # Translation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

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
    counts, _, _ = ax1.hist(
        pipeline_vectors["translation"]["len_norm_cond_prob"],
        alpha=alph,
        label="LNCP",
        color="C2",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)
    counts, _, _ = ax1.hist(
        1 - pipeline_vectors["translation"]["len_norm_entropy"],
        alpha=alph,
        label="LNE",
        color="C3",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)

    counts, _, _ = ax1.hist(
        pipeline_vectors["translation"]["multiplication_confidence"],
        alpha=alph,
        label="multiplication",
        color="C4",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)
    counts, _, _ = ax1.hist(
        pipeline_vectors["translation"]["linear_confidence"],
        alpha=alph,
        label="linear fit",
        color="C5",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)

    counts, _, _ = ax2.hist(
        translator_vectors["translation"]["weighted_semantic_density"],
        alpha=alph,
        color="C1",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)
    counts, _, _ = ax2.hist(
        translator_vectors["translation"]["len_norm_cond_prob"],
        alpha=alph,
        label="LNCP",
        color="C2",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)
    counts, _, _ = ax2.hist(
        1 - translator_vectors["translation"]["len_norm_entropy"],
        alpha=alph,
        label="LNE",
        color="C3",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)
    counts, _, _ = ax2.hist(
        translator_vectors["translation"]["comet_score"],
        alpha=alph,
        color="C0",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)

    ax2.set_xlabel("Score")
    ax1.set_ylabel("Count")
    ax2.set_ylabel("Count")
    ax1.legend(title="Metric")
    max_val = max([max(count) for count in counts_list])
    ax1.set_ylim(0, max_val)
    ax2.set_ylim(0, max_val)

    ax1.set_xlim(0, 1)
    ax2.set_xlim(0, 1)
    plt.savefig(f"{save_directory}/figures/translation_confidence_histogram.pdf")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

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
    counts, _, _ = ax1.hist(
        pipeline_vectors["classification"]["clean_confidence"],
        alpha=alph,
        color="C2",
        label="Clean Confidence",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)
    counts, _, _ = ax1.hist(
        pipeline_vectors["classification"]["multiplication_confidence"],
        alpha=alph,
        color="C3",
        label="Multiplication",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)
    counts, _, _ = ax1.hist(
        pipeline_vectors["classification"]["linear_confidence"],
        alpha=alph,
        color="C4",
        label="Linear Fit",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)
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
    counts, _, _ = ax2.hist(
        classifier_vectors["classification"]["clean_confidence"],
        alpha=alph,
        color="C2",
        bins=np.linspace(0, 1, n_bins),
    )
    counts_list.append(counts)
    ax1.legend(title="Metric")

    ax2.set_xlabel("Score")
    ax1.set_ylabel("Count")
    ax2.set_ylabel("Count")
    ax1.legend(title="Metric")
    max_val = max([max(count) for count in counts_list])
    ax1.set_ylim(0, max_val)
    ax2.set_ylim(0, max_val)

    ax1.set_xlim(0, 1)
    ax2.set_xlim(0, 1)
    plt.savefig(f"{save_directory}/figures/classification_confidence_histogram.pdf")

    ax1.set_xlim(0, 1)
    ax2.set_xlim(0, 1)
    plt.savefig(f"{save_directory}/figures/classification_confidence_histogram.pdf")
