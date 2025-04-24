import json
import math
from itertools import product

import pandas as pd
from jsonargparse import CLI
from tqdm import tqdm

from arc_spice.analysis.analysis_functions import error_propagation_analysis
from arc_spice.analysis.utils import mean_square_error


def main(experiment_path: str):
    """
    Run various combinations of confidence scores for propagation models to understand
    which combinations gives the best performance. Should produce:
      - tables for report

    Args:
        experiment_path: path to experiment directory containing:
        - full_pipeline.json
        - ocr.json
        - translator.json
        - classifier.json
    """

    save_path = f"{experiment_path}/analysis_outputs"
    metrics = {
        "recognition": [
            ("mean_confidence", "character_accuracy_rate", "ME"),
            ("mean_scores", "character_accuracy_rate", "S"),
        ],
        "translation": [
            ("weighted_semantic_density", "comet_score", "SD"),
            (
                "len_norm_cond_prob",
                "comet_score",
                "P",
            ),
            ("len_norm_confidence", "comet_score", "CE"),
        ],
        "classification": [
            ("clean_confidence", "hamming_accuracy", "CE"),
            ("mean_predicted_confidence", "hamming_accuracy", "ME"),
            ("mean_predicted_scores", "hamming_accuracy", "S"),
        ],
    }

    metric_maps = []
    metric_combinations = product(*metrics.values())
    for metric_triple in metric_combinations:
        metric_maps.append(
            {
                "recognition": tuple(metric_triple[0]),
                "translation": tuple(metric_triple[1]),
                "classification": tuple(metric_triple[2]),
            }
        )

    combinations = []
    base_scores = []
    multiplation_scores = []
    linear_scores = []
    gaussian_lin_scores = []
    gaussian_rbf_scores = []

    all_combination_results = []
    for metric_map in tqdm(metric_maps, desc="Combination"):
        # Save the combination of metrics
        combinations.append(tuple(values[2] for values in metric_map.values()))

        pipeline_vectors, _, _ = error_propagation_analysis(
            experiment_path, metric_map=metric_map
        )
        confidence_targets = [
            "character_accuracy_rate",
            "comet_score",
            "hamming_accuracy",
        ]
        confidence_metrics = [values[0] for values in metric_map.values()]
        steps = [
            "recognition",
            "translation",
            "classification",
        ]
        propagated_metrics = [
            "multiplication_confidence",
            "linear_confidence",
            "gaussian_lin_confidence",
            "gaussian_rbf_confidence",
        ]

        combination_scores = {}
        for step, target, metric in zip(
            steps, confidence_targets, confidence_metrics, strict=True
        ):
            combination_scores[step] = {}
            combination_scores[step][metric] = mean_square_error(
                predicted=pipeline_vectors[step][metric],
                error=pipeline_vectors[step][target],
            )
            for prop_metric in propagated_metrics:
                combination_scores[step][prop_metric] = mean_square_error(
                    predicted=pipeline_vectors[step][prop_metric],
                    error=pipeline_vectors[step][target],
                )

        all_combination_results.append(combination_scores)
        b_scores = tuple(
            round(math.sqrt(combination_scores[step][metric_map[step][0]]), 3)
            for step in steps
        )
        base_scores.append((round(sum(b_scores) / len(b_scores), 3), b_scores[-1]))
        m_scores = tuple(
            round(
                math.sqrt(combination_scores[step]["multiplication_confidence"]),
                3,
            )
            for step in steps
        )
        multiplation_scores.append(
            (round(sum(m_scores) / len(m_scores), 3), m_scores[-1])
        )
        l_scores = tuple(
            round(math.sqrt(combination_scores[step]["linear_confidence"]), 3)
            for step in steps
        )
        linear_scores.append((round(sum(l_scores) / len(l_scores), 3), l_scores[-1]))
        g_l_scores = tuple(
            round(math.sqrt(combination_scores[step]["gaussian_lin_confidence"]), 3)
            for step in steps
        )
        gaussian_lin_scores.append(
            (round(sum(g_l_scores) / len(g_l_scores), 3), g_l_scores[-1])
        )
        g_r_scores = tuple(
            round(math.sqrt(combination_scores[step]["gaussian_rbf_confidence"]), 3)
            for step in steps
        )
        gaussian_rbf_scores.append(
            (round(sum(g_r_scores) / len(g_r_scores), 3), g_r_scores[-1])
        )

    propagation_data = {
        "Combination": combinations,
        "Base": base_scores,
        "Multiplication": multiplation_scores,
        "Linear Fit": linear_scores,
        "Gaussian (Linear)": gaussian_lin_scores,
        "Gaussian (RBF)": gaussian_rbf_scores,
    }

    propagation_dataframe = pd.DataFrame(data=propagation_data)
    with open(
        f"{save_path}/tables/propagation_combinations_new.tex", "w+"
    ) as table_file:
        table_file.write(
            propagation_dataframe.to_latex(
                index=False,
                label="tab:propagation_combinations",
                caption=(
                    "RMSE Propagation Combinations for various confience metrics,"
                    " left value is the average RMSE across all steps, right value"
                    " is the RMSE for the classification step."
                ),
            )
        )

    with open(f"{save_path}/raw_results/propagation_combinations.json", "w") as f:
        json.dump(all_combination_results, f, indent=2)


if __name__ == "__main__":
    CLI(main)
