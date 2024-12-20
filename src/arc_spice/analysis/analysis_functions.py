from arc_spice.analysis.utils import fitted_uq_model, multiplication_prop
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


def error_propagation_analysis(
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
    step_keys = ["recognition", "translation", "classification"]

    pipeline_results = open_json_path(f"{experiment_path}/full_pipeline.json")

    vectors_w_lin, celex_ids = fitted_uq_model(
        results_dict=exp_vectors(pipeline_results, step_keys)
    )
    pipeline_vectors = exp_vectors(
        pipeline_results, step_keys, target_celex_ids=celex_ids["test_ids"]
    )
    multiplication_vectors = multiplication_prop(pipeline_vectors)
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
