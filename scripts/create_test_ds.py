import os
import shutil

import datasets
from datasets import load_dataset

from arc_spice.data import multieurlex_utils

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTDATA_DIR = os.path.join(PROJECT_ROOT, "tests/testdata")
BASE_DATASET_INFO_MULTILANG = os.path.join(
    TESTDATA_DIR, "base_testdata/dataset_info.json"
)
BASE_DATASET_INFO_EN = os.path.join(TESTDATA_DIR, "base_testdata/dataset_info_en.json")
BASE_DATASET_METADATA_DIR = os.path.join(TESTDATA_DIR, "base_testdata/MultiEURLEX")

# TODO
CONTENT_MULTILANG: list[dict[str, str]] = [
    {
        "en": f"Some text before the marker 1 {multieurlex_utils.ARTICLE_1_MARKERS['en']} Some text after the marker 1",  # noqa: E501
        "fr": f"Some text before the marker 1 {multieurlex_utils.ARTICLE_1_MARKERS['fr']} Some text after the marker 1",  # noqa: E501
        "de": f"Some text before the marker 1 {multieurlex_utils.ARTICLE_1_MARKERS['de']} Some text after the marker 1",  # noqa: E501
    },
    {
        "en": f"Some text before the marker 2 {multieurlex_utils.ARTICLE_1_MARKERS['en']} Some text after the marker 2",  # noqa: E501
        "fr": f"Some text before the marker 2 {multieurlex_utils.ARTICLE_1_MARKERS['fr']} Some text after the marker 2",  # noqa: E501
        "de": f"Some text before the marker 2 {multieurlex_utils.ARTICLE_1_MARKERS['de']} Some text after the marker 2",  # noqa: E501
    },
    {
        "en": "Some text before the marker 3",  # no marker, no text after marker
        "fr": "Some text before the marker 3",
        "de": "Some text before the marker 3",
    },
    {
        "en": f"Some text before the marker 4 {multieurlex_utils.ARTICLE_1_MARKERS['en']} Some text after the marker 4",  # noqa: E501
        "fr": f"Some text before the marker 4 {multieurlex_utils.ARTICLE_1_MARKERS['fr']} Some text after the marker 4",  # noqa: E501
        "de": f"Some text before the marker 4 {multieurlex_utils.ARTICLE_1_MARKERS['de']} Some text after the marker 4",  # noqa: E501
    },
    {
        "en": f"Some text before the marker 5 {multieurlex_utils.ARTICLE_1_MARKERS['en']} Some text after the marker 5",  # noqa: E501
        "fr": f"Some text before the marker 5 {multieurlex_utils.ARTICLE_1_MARKERS['fr']} Some text after the marker 5",  # noqa: E501
        "de": f"Some text before the marker 5 {multieurlex_utils.ARTICLE_1_MARKERS['de']} Some text after the marker 5",  # noqa: E501
    },
]
CONTENT_EN: list[str] = [
    f"Some text before the marker 1 {multieurlex_utils.ARTICLE_1_MARKERS['en']} Some text after the marker 1",  # noqa: E501
    f"Some text before the marker 2 {multieurlex_utils.ARTICLE_1_MARKERS['en']} Some text after the marker 2",  # noqa: E501
    "Some text before the marker 3",  # no marker, no text after marker
    f"Some text before the marker 4 {multieurlex_utils.ARTICLE_1_MARKERS['en']} Some text after the marker 4",  # noqa: E501
    f"Some text before the marker 5 {multieurlex_utils.ARTICLE_1_MARKERS['en']} Some text after the marker 5",  # noqa: E501
]


def overwrite_text(
    _orig,
    i: int,
    content: list[dict[str, str]] | list[str],
) -> dict[str, str | dict[str, str]]:
    return {"text": content[i]}


def create_test_ds(
    testdata_dir: str,
    ds_name: str,
    content: list[dict[str, str]] | list[str],
    dataset_info_fpath: str,
) -> None:
    dataset = load_dataset(
        "multi_eurlex",
        "all_languages",
        label_level="level_1",
        trust_remote_code=True,
    )

    dataset["train"] = dataset["train"].take(5)
    dataset["validation"] = dataset["validation"].take(5)
    dataset["test"] = dataset["test"].take(5)

    dataset = dataset.map(
        overwrite_text,
        with_indices=True,
        fn_kwargs={"content": content},
    )

    dataset.save_to_disk(os.path.join(testdata_dir, ds_name))

    shutil.copy(
        dataset_info_fpath,
        os.path.join(testdata_dir, ds_name, "train/dataset_info.json"),
    )
    shutil.copy(
        dataset_info_fpath,
        os.path.join(testdata_dir, ds_name, "validation/dataset_info.json"),
    )
    shutil.copy(
        dataset_info_fpath,
        os.path.join(testdata_dir, ds_name, "test/dataset_info.json"),
    )
    # metadata copy
    shutil.copytree(
        BASE_DATASET_METADATA_DIR,
        os.path.join(testdata_dir, ds_name, "MultiEURLEX"),
    )

    assert datasets.load_from_disk(os.path.join(testdata_dir, ds_name)) is not None


if __name__ == "__main__":
    os.makedirs(TESTDATA_DIR, exist_ok=True)

    content = [
        "Some text before the marker en Some text after the marker",
        "Some text before the marker fr Some text after the marker",
    ]

    create_test_ds(
        testdata_dir=TESTDATA_DIR,
        ds_name="multieurlex_test",
        content=CONTENT_MULTILANG,
        dataset_info_fpath=BASE_DATASET_INFO_MULTILANG,
    )

    create_test_ds(
        testdata_dir=TESTDATA_DIR,
        ds_name="multieurlex_test_en",
        content=CONTENT_EN,
        dataset_info_fpath=BASE_DATASET_INFO_EN,
    )
