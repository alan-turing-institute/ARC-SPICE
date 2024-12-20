from jsonargparse import CLI

from arc_spice.analysis.analysis_functions import (
    error_propagation_analysis,
    single_model_analysis,
)


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
    pipeline_vectors, translation_vectors, classifier_vectors = (
        error_propagation_analysis(experiment_path)
    )
    # THESE ARE ALL IN CORRECT ORDER NOW
    # Below code should include:
    #   - brier score for all vectors
    #   - distributional plots
    #   - tables for report

    single_model_brier_scores = single_model_analysis(experiment_path)
    print(len(pipeline_vectors), len(translation_vectors), len(classifier_vectors))
    print(single_model_brier_scores)


if __name__ == "__main__":
    CLI(main)
