import json

from jsonargparse import CLI

from arc_spice.eval.analysis_utils import exp_analysis
from arc_spice.utils import open_json_path


def main(
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
    experiment_results_dict = {
        "individual_components": ind_results,
        "pipeline": pipe_results,
    }

    # save results
    with open(f"{experiment_path}/analysis_results.json", "w") as save_file:
        json.dump(experiment_results_dict, save_file, indent=2)


if __name__ == "__main__":
    CLI(main)
