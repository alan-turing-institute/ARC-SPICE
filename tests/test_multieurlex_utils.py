import os
from typing import Any
from unittest.mock import patch

import datasets
import numpy as np
import pyarrow as pa
import pytest
from datasets.formatting import PythonFormatter
from datasets.formatting.formatting import LazyRow
from PIL import Image as PILImage

from arc_spice.data import multieurlex_utils

TEST_ROOT = os.path.dirname(os.path.abspath(__file__))


def _create_single_lang_row(text) -> LazyRow:
    pa_table = pa.Table.from_pydict({"text": [text]})
    formatter = PythonFormatter(lazy=True)
    return formatter.format_row(pa_table)


def _create_multilang_row(texts_by_lang: dict[str, str]) -> LazyRow:
    d = [{"text": texts_by_lang}]
    pa_table = pa.Table.from_pylist(d)
    formatter = PythonFormatter(lazy=True)
    return formatter.format_row(pa_table)


def _create_translation_row(source_text: str, target_text: str) -> LazyRow:
    pa_table = pa.Table.from_pydict(
        {"source_text": [source_text], "target_text": [target_text]}
    )
    formatter = PythonFormatter(lazy=True)
    return formatter.format_row(pa_table)


def test_extract_articles_single_lang():
    langs = ["en"]
    pre_text = "Some text before the marker"
    post_text = "Some text after the marker"
    row = _create_single_lang_row(
        text=f"{pre_text} {multieurlex_utils.ARTICLE_1_MARKERS['en']} {post_text}"
    )
    out = multieurlex_utils.extract_articles(item=row, languages=langs)
    assert out == {"text": f"{multieurlex_utils.ARTICLE_1_MARKERS['en']} {post_text}"}


def test_extract_articles_multi_lang():
    langs = ["en", "fr"]
    pre_text = "Some text before the marker"
    post_text = "Some text after the marker"
    texts = {
        lang: f"{pre_text} {multieurlex_utils.ARTICLE_1_MARKERS[lang]} {post_text}"
        for lang in langs
    }
    row = _create_multilang_row(texts_by_lang=texts)
    out = multieurlex_utils.extract_articles(item=row, languages=langs)
    assert out == {
        "text": {
            "en": f"{multieurlex_utils.ARTICLE_1_MARKERS['en']} {post_text}",
            "fr": f"{multieurlex_utils.ARTICLE_1_MARKERS['fr']} {post_text}",
        }
    }


def test_make_ocr_data():
    source_text = "Some text to make into an image"
    row = _create_translation_row(source_text=source_text, target_text="foo")
    dummy_im1 = PILImage.fromarray(
        np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8)
    )
    dummy_im2 = PILImage.fromarray(
        np.random.randint(0, 255, (10, 10, 3)).astype(np.uint8)
    )

    with patch("arc_spice.data.multieurlex_utils.GeneratorFromStrings") as mock_gen:
        mock_gen.return_value = [(dummy_im1, "target1"), (dummy_im2, "target2")]
        output = multieurlex_utils.make_ocr_data(row)

    assert output == {
        "ocr_data": {
            "ocr_images": (dummy_im1, dummy_im2),
            "ocr_targets": ("target1", "target2"),
        }
    }


def _check_keys_untouched(
    original_dataset: datasets.Dataset,
    dataset: datasets.Dataset,
    indices_kept: list[int],
    ignore_keys=list[str],
) -> None:
    # check remaining keys are untouched
    for key in dataset.features:
        if key not in ignore_keys:
            assert dataset[key] == [original_dataset[key][i] for i in indices_kept]


def test_load_multieurlex_en():
    data_dir = f"{TEST_ROOT}/testdata/multieurlex_test_en"
    level = 1
    languages = ["en"]
    drop_empty = True

    ds = datasets.load_from_disk(data_dir)
    expected_keys = {"celex_id", "text", "labels"}
    expected_non_empty_indices = [0, 1, 3, 4]
    text_expected_non_empty_indices = [i + 1 for i in expected_non_empty_indices]
    with patch(
        "arc_spice.data.multieurlex_utils.datasets.load_dataset", return_value=ds
    ):
        dataset_dict, metadata = multieurlex_utils.load_multieurlex(
            data_dir=data_dir, level=level, languages=languages, drop_empty=drop_empty
        )
        assert len(dataset_dict) == 3
        for split in ["train", "validation", "test"]:
            assert set(dataset_dict[split].features.keys()) == expected_keys
            assert len(dataset_dict[split]) == 4  # 5 items, 1 is empty so dropped
            assert dataset_dict[split]["text"] == [
                f"{multieurlex_utils.ARTICLE_1_MARKERS['en']} Some text after the marker {i}"  # noqa: E501
                for i in text_expected_non_empty_indices  # 3 dropped
            ]
            _check_keys_untouched(
                original_dataset=ds[split],
                dataset=dataset_dict[split],
                indices_kept=expected_non_empty_indices,
                ignore_keys=["text"],
            )


def test_load_multieurlex_multi_lang():
    data_dir = f"{TEST_ROOT}/testdata/multieurlex_test"
    level = 1
    languages = ["de", "en", "fr"]
    drop_empty = True

    ds = datasets.load_from_disk(data_dir)
    expected_keys = {"celex_id", "text", "labels"}
    expected_non_empty_indices = [0, 1, 3, 4]
    text_expected_non_empty_indices = [i + 1 for i in expected_non_empty_indices]
    with patch(
        "arc_spice.data.multieurlex_utils.datasets.load_dataset", return_value=ds
    ):
        dataset_dict, metadata = multieurlex_utils.load_multieurlex(
            data_dir=data_dir, level=level, languages=languages, drop_empty=drop_empty
        )
        assert len(dataset_dict) == 3
        for split in ["train", "validation", "test"]:
            assert set(dataset_dict[split].features.keys()) == expected_keys
            assert len(dataset_dict[split]) == 4  # 5 items, 1 is empty so dropped
            assert dataset_dict[split]["text"] == [  #
                {
                    lang: f"{multieurlex_utils.ARTICLE_1_MARKERS[lang]} Some text after the marker {i}"  # noqa: E501
                    for lang in languages
                }
                for i in text_expected_non_empty_indices  # 3 dropped
            ]
            _check_keys_untouched(
                original_dataset=ds[split],
                dataset=dataset_dict[split],
                indices_kept=expected_non_empty_indices,
                ignore_keys=["text"],
            )


def _check_pipeline_text(
    dataset: datasets.Dataset,
    text_indices_kept: list[int],
    source_lang: str,
    target_lang: str,
):
    assert dataset["source_text"] == [
        f"{multieurlex_utils.ARTICLE_1_MARKERS[source_lang]} Some text after the marker {i}"  # noqa: E501
        for i in text_indices_kept
    ]
    assert dataset["target_text"] == [
        f"{multieurlex_utils.ARTICLE_1_MARKERS[target_lang]} Some text after the marker {i}"  # noqa: E501
        for i in text_indices_kept
    ]


def _test_load_multieurlex_for_pipeline(expected_keys: set[str], load_ocr_data: bool):
    data_dir = f"{TEST_ROOT}/testdata/multieurlex_test"
    lang_pair = {"source": "de", "target": "en"}

    ds = datasets.load_from_disk(data_dir)

    expected_non_empty_indices = [0, 1, 3, 4]
    text_expected_non_empty_indices = [i + 1 for i in expected_non_empty_indices]
    with patch(
        "arc_spice.data.multieurlex_utils.datasets.load_dataset", return_value=ds
    ):
        dataset_dict, metadata = multieurlex_utils.load_multieurlex_for_pipeline(
            data_dir=data_dir,
            level=1,
            lang_pair=lang_pair,
            drop_empty=True,
            load_ocr_data=load_ocr_data,
        )
    assert len(dataset_dict) == 3
    for split in ["train", "validation", "test"]:
        assert set(dataset_dict[split].features.keys()) == expected_keys
        assert len(dataset_dict[split]) == 4  # 5 items, 1 is empty so dropped
        _check_pipeline_text(
            dataset=dataset_dict[split],
            text_indices_kept=text_expected_non_empty_indices,  # inds start at 1
            source_lang=lang_pair["source"],
            target_lang=lang_pair["target"],
        )
        _check_keys_untouched(
            original_dataset=ds[split],
            dataset=dataset_dict[split],
            indices_kept=expected_non_empty_indices,  # inds start at 0
            ignore_keys=["source_text", "target_text", "ocr_data"],
        )

    return dataset_dict, metadata


def test_load_multieurlex_for_pipeline():
    expected_keys = {"celex_id", "labels", "source_text", "target_text"}
    _test_load_multieurlex_for_pipeline(
        expected_keys=expected_keys, load_ocr_data=False
    )


def _create_ocr_data(
    expected_n_rows: int,
) -> tuple[list[dict[str, Any]], tuple[list[PILImage.Image]], tuple[list[str]]]:
    dummy_ocr_data = [
        {
            "ocr_images": [PILImage.fromarray(np.ones((5, 5, 3)).astype(np.uint8) * i)]
            * (i + 1),
            "ocr_targets": [f"foo {i}"] * (i + 1),
        }
        for i in range(expected_n_rows)
    ]  # different value at each call, different number of items each time (nonzero i+1)

    # unpack data to give expected valuess
    expected_ocr_data, expected_ocr_targets = zip(
        *[x.values() for x in dummy_ocr_data], strict=True
    )
    return dummy_ocr_data, expected_ocr_data, expected_ocr_targets


# same as above but with OCR data checks
# TO FIX
@pytest.mark.skip(
    reason=(
        "This is currently broken by changes to: "
        "_make_ocr_data and load_multieurlex_for_pipeline"
    )
)
def test_load_multieurlex_for_pipeline_ocr():
    expected_keys = {
        "celex_id",
        "labels",
        "source_text",
        "target_text",
        "ocr_data",
    }

    expected_n_rows = 4
    dummy_ocr_data, expected_ocr_data, expected_ocr_targets = _create_ocr_data(
        expected_n_rows=expected_n_rows
    )

    with patch("arc_spice.data.multieurlex_utils.make_ocr_data") as mock_mod:
        mock_mod.side_effect = dummy_ocr_data * 3  # handle all 3 splits
        dataset_dict, metadata = _test_load_multieurlex_for_pipeline(
            expected_keys=expected_keys, load_ocr_data=True
        )
    for split in ["train", "validation", "test"]:
        for row_index in range(len(dataset_dict[split])):
            # OCR - images
            #  PIL.PngImagePlugin.PngImageFile vs PIL.Image.Image so compare as np
            output_as_numpy = [
                np.asarray(im)
                for im in dataset_dict[split]["ocr_data"]["ocr_images"][row_index]
            ]
            expected_as_numpy = [np.asarray(im) for im in expected_ocr_data[row_index]]
            np.testing.assert_array_equal(output_as_numpy, expected_as_numpy)
            # OCR - targets
            assert (
                dataset_dict[split]["ocr_data"]["ocr_targets"][row_index]
                == expected_ocr_targets[row_index]
            )
