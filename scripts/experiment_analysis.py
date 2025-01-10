import json
import os

import pandas as pd
from jsonargparse import CLI

from arc_spice.analysis.analysis_functions import (
    error_propagation_analysis,
    plot_vectors,
    single_model_analysis,
)
from arc_spice.analysis.utils import mean_square_error


def main(
    experiment_path: str,
):
    """
    Run analysis on a given experiment with provided experiment path, should produce:
      - distributional plots
      - tables for report

    Args:
        experiment_path: path to experiment directory containing:
        - full_pipeline.json
        - ocr.json
        - translator.json
        - classifier.json
    """
    save_path = f"{experiment_path}/analysis_outputs"
    os.makedirs(name=f"{save_path}/figures", exist_ok=True)
    os.makedirs(name=f"{save_path}/tables", exist_ok=True)
    os.makedirs(name=f"{save_path}/raw_results", exist_ok=True)

    pipeline_vectors, translation_vectors, classification_vectors = (
        error_propagation_analysis(experiment_path)
    )
    plot_vectors(
        save_directory=save_path,
        pipeline_vectors=pipeline_vectors,
        translator_vectors=translation_vectors,
        classifier_vectors=classification_vectors,
    )

    confidence_targets = ["character_accuracy_rate", "comet_score", "hamming_accuracy"]
    confidence_metrics = [
        "mean_confidence",
        "weighted_semantic_density",
        "mean_predicted_confidence",
    ]
    steps = ["recognition", "translation", "classification"]
    propagated_metrics = [
        "multiplication_confidence",
        "linear_confidence",
        "gaussian_confidence",
    ]
    additional_metrics = {
        "recognition": [],
        "translation": ["len_norm_cond_prob", "len_norm_confidence"],
        "classification": ["clean_confidence"],
    }

    pipeline_scores = {}
    for step, target, metric in zip(
        steps, confidence_targets, confidence_metrics, strict=True
    ):
        pipeline_scores[step] = {}
        pipeline_scores[step][metric] = mean_square_error(
            predicted=pipeline_vectors[step][metric],
            error=pipeline_vectors[step][target],
        )
        for prop_metric in propagated_metrics:
            pipeline_scores[step][prop_metric] = mean_square_error(
                predicted=pipeline_vectors[step][prop_metric],
                error=pipeline_vectors[step][target],
            )
        for additional_metric in additional_metrics[step]:
            pipeline_scores[step][additional_metric] = mean_square_error(
                predicted=pipeline_vectors[step][additional_metric],
                error=pipeline_vectors[step][target],
            )

    with open(f"{save_path}/raw_results/propagation_mean_square_errors.json", "w") as f:
        json.dump(pipeline_scores, f, indent=2)

    base_scores = [
        pipeline_scores["recognition"]["mean_confidence"],
        pipeline_scores["translation"]["weighted_semantic_density"],
        pipeline_scores["classification"]["mean_predicted_confidence"],
    ]
    multiplication_scores = [
        pipeline_scores[step]["multiplication_confidence"] for step in steps
    ]
    linear_scores = [pipeline_scores[step]["linear_confidence"] for step in steps]
    gaussian_scores = [pipeline_scores[step]["gaussian_confidence"] for step in steps]
    propagation_data = {
        "Steps": steps,
        "Base Score": base_scores,
        "Multiplication Score": multiplication_scores,
        "Linear Score": linear_scores,
        "Gaussian Scores": gaussian_scores,
    }

    propagation_dataframe = pd.DataFrame(data=propagation_data)
    with open(f"{save_path}/tables/propagation_comparison.tex", "w+") as table_file:
        table_file.write(propagation_dataframe.to_latex(index=False))

    # single model error analysis
    error_analysis = single_model_analysis(experiment_path)

    with open(f"{save_path}/raw_results/error_analysis.json", "w") as f:
        json.dump(error_analysis, f, indent=2)

    recogntion_errors = [
        error_analysis["individual_components"]["ocr"]["mean_accuracy"],
        error_analysis["pipeline"]["ocr"]["mean_accuracy"],
    ]
    translation_errors = [
        error_analysis["individual_components"]["translator"]["mean_accuracy"],
        error_analysis["pipeline"]["translator"]["mean_accuracy"],
    ]
    recognition_errors = [
        error_analysis["individual_components"]["classifier"]["mean_accuracy"],
        error_analysis["pipeline"]["classifier"]["mean_accuracy"],
    ]
    base_errors_data = {
        "Usage": ["Base", "Pipeline"],
        "Recognition (Character Accuracy Rate)": recogntion_errors,
        "Translation (Comet Score)": translation_errors,
        "Classification (Hamming Accuracy)": recognition_errors,
    }
    base_errors_dataframe = pd.DataFrame(data=base_errors_data)
    with open(f"{save_path}/tables/base_errors.tex", "w+") as table_file:
        table_file.write(base_errors_dataframe.to_latex(index=False))

    # custom metric map example

    # THIS IS CURRENTLY NOT WORKING AS EXPECTED #

    metric_map = {
        "recognition": ("mean_confidence", "character_accuracy_rate"),
        "translation": ("len_norm_cond_prob", "comet_score"),
        "classification": ("clean_confidence", "hamming_accuracy"),
    }

    pipeline_vectors, _, _ = error_propagation_analysis(
        experiment_path, metric_map=metric_map
    )

    confidence_targets = ["character_accuracy_rate", "comet_score", "hamming_accuracy"]
    confidence_metrics = [values[0] for values in metric_map.values()]
    steps = ["recognition", "translation", "classification"]
    propagated_metrics = [
        "multiplication_confidence",
        "linear_confidence",
        "gaussian_confidence",
    ]

    pipeline_scores = {}
    for step, target, metric in zip(
        steps, confidence_targets, confidence_metrics, strict=True
    ):
        pipeline_scores[step] = {}
        pipeline_scores[step][metric] = mean_square_error(
            predicted=pipeline_vectors[step][metric],
            error=pipeline_vectors[step][target],
        )
        for prop_metric in propagated_metrics:
            pipeline_scores[step][prop_metric] = mean_square_error(
                predicted=pipeline_vectors[step][prop_metric],
                error=pipeline_vectors[step][target],
            )

    with open(
        f"{save_path}/raw_results/custom_propagation_mean_square_errors.json", "w"
    ) as f:
        json.dump(pipeline_scores, f, indent=2)


if __name__ == "__main__":
    CLI(main)
