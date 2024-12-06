import os
from unittest.mock import MagicMock, patch

import pytest

from arc_spice.utils import open_yaml_path
from arc_spice.variational_pipelines.RTC_single_component_pipeline import (
    ClassificationVariationalPipeline,
    RecognitionVariationalPipeline,
    TranslationVariationalPipeline,
)
from arc_spice.variational_pipelines.RTC_variational_pipeline import (
    RTCVariationalPipeline,
)

CONFIG_ROOT = f"{os.path.dirname(os.path.abspath(__file__))}/../config/"

PIPELINE_PATH = f"{CONFIG_ROOT}/RTC_configs/roberta-mt5-zero-shot.yaml"


@pytest.fixture()
def dummy_data():
    return {
        "ocr_data": [[0.1, 0.2], [0.2, 0.3]],
        "source_text": "source text",
        "target_text": "target text",
        "labels": [0, 1],
    }


@pytest.fixture()
def dummy_metadata():
    n_classes = 5
    return {
        "n_classes": n_classes,
        "class_labels": list(range(n_classes)),
        "class_descriptors": [
            {"en": f"class_{i}", "fr": f"classe_{i}"} for i in list(range(n_classes))
        ],
    }


def test_pipeline_inputs(dummy_data, dummy_metadata):
    pipeline_config = open_yaml_path(PIPELINE_PATH)

    with patch(  # noqa: SIM117
        "arc_spice.variational_pipelines.RTC_variational_pipeline.pipeline",
        return_value=None,
    ):
        with patch(
            (
                "arc_spice.variational_pipelines.RTC_variational_pipeline."
                "RTCVariationalPipeline._init_semantic_density"
            ),
            return_value=None,
        ):
            pipeline = RTCVariationalPipeline(
                model_pars=pipeline_config,
                data_pars=dummy_metadata,
                translation_batch_size=1,
            )

    dummy_recognise_output = {"outputs": "rec text"}
    dummy_translate_output = {"outputs": ["translate text"]}
    dummy_classification_output = {"outputs": "classification"}

    pipeline.recognise = MagicMock(return_value=dummy_recognise_output)
    pipeline.translate = MagicMock(return_value=dummy_translate_output)
    pipeline.classify_topic = MagicMock(return_value=dummy_classification_output)

    pipeline.clean_inference(dummy_data)

    pipeline.recognise.assert_called_with(dummy_data)
    pipeline.translate.assert_called_with("rec text")
    pipeline.classify_topic.assert_called_with("translate text")


def test_single_component_inputs(dummy_data, dummy_metadata):
    pipeline_config = open_yaml_path(PIPELINE_PATH)
    dummy_recognise_output = {"outputs": "rec text"}
    dummy_translate_output = {"outputs": ["translate text"]}
    dummy_classification_output = {"outputs": "classification"}

    with patch(  # noqa: SIM117
        "arc_spice.variational_pipelines.RTC_single_component_pipeline.pipeline"
    ):
        with patch(
            (
                "arc_spice.variational_pipelines.RTC_single_component_pipeline."
                "RTCSingleComponentPipeline._init_semantic_density"
            ),
            return_value=None,
        ):
            recognise_pipeline = RecognitionVariationalPipeline(
                model_pars=pipeline_config,
            )
            translate_pipeline = TranslationVariationalPipeline(
                model_pars=pipeline_config,
                translation_batch_size=1,
            )
            classify_pipeline = ClassificationVariationalPipeline(
                model_pars=pipeline_config,
                data_pars=dummy_metadata,
            )

    recognise_pipeline.forward_function = MagicMock(return_value=dummy_recognise_output)
    translate_pipeline.forward_function = MagicMock(return_value=dummy_translate_output)
    classify_pipeline.forward_function = MagicMock(
        return_value=dummy_classification_output
    )

    recognise_pipeline.clean_inference(dummy_data)
    translate_pipeline.clean_inference(dummy_data)
    classify_pipeline.clean_inference(dummy_data)

    recognise_pipeline.forward_function.assert_called_with(dummy_data["ocr_data"])
    translate_pipeline.forward_function.assert_called_with(dummy_data["source_text"])
    classify_pipeline.forward_function.assert_called_with(dummy_data["target_text"])
