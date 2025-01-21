import json
import math
from itertools import product

import numpy as np
from jsonargparse import CLI
from sklearn.model_selection import KFold
from tqdm import tqdm

from arc_spice.analysis.analysis_functions import (
    k_fold_error_propagation_analysis,
    plot_error_propagation,
)
from arc_spice.analysis.utils import mean_square_error
from arc_spice.utils import open_json_path


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
    n_rows = len(open_json_path(f"{experiment_path}/full_pipeline.json"))
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
    k = 4

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
    results = {}
    all_combinations = []
    for metric_map in metric_maps:
        comb_key = str(tuple(values[2] for values in metric_map.values()))
        all_combinations.append(comb_key)
        results[comb_key] = {
            "recognition": {
                metric_map["recognition"][0]: [],
                "multiplication_confidence": [],
                "linear_confidence": [],
            },
            "translation": {
                metric_map["translation"][0]: [],
                "multiplication_confidence": [],
                "linear_confidence": [],
            },
            "classification": {
                metric_map["classification"][0]: [],
                "multiplication_confidence": [],
                "linear_confidence": [],
            },
        }

    kf = KFold(n_splits=k, shuffle=True)
    splits = kf.split(np.arange(n_rows))

    for split_indices in tqdm(splits, desc="Split", total=k):
        split_dict = {"train": split_indices[1], "test": split_indices[0]}
        for metric_map in metric_maps:
            # Save the combination of metrics
            combination_key = str(tuple(values[2] for values in metric_map.values()))

            pipeline_vectors, _, _ = k_fold_error_propagation_analysis(
                experiment_path, metric_map=metric_map, splits=split_dict
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
            ]

            for step, target, metric in zip(
                steps, confidence_targets, confidence_metrics, strict=True
            ):
                results[combination_key][step][metric].append(
                    math.sqrt(
                        mean_square_error(
                            predicted=pipeline_vectors[step][metric],
                            error=pipeline_vectors[step][target],
                        )
                    )
                )
                for prop_metric in propagated_metrics:
                    results[combination_key][step][prop_metric].append(
                        math.sqrt(
                            mean_square_error(
                                predicted=pipeline_vectors[step][prop_metric],
                                error=pipeline_vectors[step][target],
                            )
                        )
                    )

    with open(f"{save_path}/raw_results/k_fold_propagation.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_error_propagation(
        results, save_directory=save_path, combination_keys=all_combinations
    )


if __name__ == "__main__":
    CLI(main)
