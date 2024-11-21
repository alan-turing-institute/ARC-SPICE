import os
from unittest.mock import patch

import datasets
import pyarrow as pa
from datasets.formatting import PythonFormatter
from datasets.formatting.formatting import LazyRow

from arc_spice.data import multieurlex_utils

# def extract_articles(
#     item: LazyRow, languages: list[str]
# ) -> dict[str, str] | dict[str, dict[str, str]]:

TEST_ROOT = os.path.dirname(os.path.abspath(__file__))


def _create_row(text) -> LazyRow:
    pa_table = pa.Table.from_pydict({"text": [text]})
    formatter = PythonFormatter(lazy=True)
    return formatter.format_row(pa_table)


def _create_multilang_row(texts_by_lang: dict[str, str]) -> LazyRow:
    d = [{"text": texts_by_lang}]
    pa_table = pa.Table.from_pylist(d)
    formatter = PythonFormatter(lazy=True)
    return formatter.format_row(pa_table)


def test_extract_articles_single_lang():
    langs = ["en"]
    pre_text = "Some text before the marker"
    post_text = "Some text after the marker"
    row = _create_row(
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


def test_load_multieurlex_en():
    data_dir = f"{TEST_ROOT}/testdata/multieurlex_test_en"
    level = 1
    languages = ["en"]
    drop_empty = True

    ds = datasets.load_from_disk(data_dir)
    with patch("arc_spice.data.multieurlex_utils.load_dataset", return_value=ds):
        dataset_dict, metadata = multieurlex_utils.load_multieurlex(
            data_dir=data_dir, level=level, languages=languages, drop_empty=drop_empty
        )
        assert len(dataset_dict) == 3
        assert len(dataset_dict["train"]) == 4  # 5 items, 1 is empty so dropped
        assert len(dataset_dict["validation"]) == 4  # 5 items, 1 is empty so dropped
        assert len(dataset_dict["test"]) == 4  # 5 items, 1 is empty so dropped
        assert dataset_dict["train"]["text"] == [
            f"{multieurlex_utils.ARTICLE_1_MARKERS["en"]} Some text after the marker {i}"  # noqa: E501
            for i in [1, 2, 4, 5]  # 3 dropped
        ]


def test_load_multieurlex_for_translation():
    data_dir = f"{TEST_ROOT}/testdata/multieurlex_test"
    level = 1
    languages = ["de", "en", "fr"]
    drop_empty = True

    ds = datasets.load_from_disk(data_dir)
    with patch("arc_spice.data.multieurlex_utils.load_dataset", return_value=ds):
        dataset_dict, metadata = multieurlex_utils.load_multieurlex(
            data_dir=data_dir, level=level, languages=languages, drop_empty=drop_empty
        )
        assert len(dataset_dict) == 3
        assert len(dataset_dict["train"]) == 4  # 5 items, 1 is empty so dropped
        assert len(dataset_dict["validation"]) == 4  # 5 items, 1 is empty so dropped
        assert len(dataset_dict["test"]) == 4  # 5 items, 1 is empty so dropped
        assert dataset_dict["train"]["text"] == [  #
            {
                lang: f"{multieurlex_utils.ARTICLE_1_MARKERS[lang]} Some text after the marker {i}"  # noqa: E501
                for lang in languages
            }
            for i in [1, 2, 4, 5]  # 3 dropped
        ]
