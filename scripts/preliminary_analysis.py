import json

from jsonargparse import CLI

from arc_spice.analysis.analysis_functions import single_model_analysis


def main(
    experiment_path: str,
):
    # save results
    with open(f"{experiment_path}/analysis_results.json", "w") as save_file:
        json.dump(single_model_analysis(experiment_path), save_file, indent=2)


if __name__ == "__main__":
    CLI(main)
