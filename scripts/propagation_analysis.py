"""
Run analysis on the different models for propagation.
"""

import json

from jsonargparse import CLI

from arc_spice.analysis.analysis_functions import propagation_analysis


def main(experiment_path: str):
    """Run analysis of a given pipeline experiment using the different propagation
    models

    Args:
        experiment_path: path experiment directory
    """
    out_res = propagation_analysis(experiment_path)
    # save results
    with open(f"{experiment_path}/prop_model_analysis.json", "w") as save_file:
        json.dump(out_res, save_file, indent=2)


if __name__ == "__main__":
    CLI(main)
