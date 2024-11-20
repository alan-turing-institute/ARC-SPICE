import json
from typing import Any

import torch
from datasets import DatasetDict, load_dataset
from datasets.formatting.formatting import LazyRow
from torch.nn.functional import one_hot

# For identifying where the adopted decisions begin
ARTICLE_1_MARKERS = {"en": "\nArticle 1\n", "fr": "\nArticle premier\n"}


# creates a multi-hot vector for classification loss
class MultiHot:
    """Class that will multi-hot encode the classes for classification."""

    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes

    def __call__(self, class_labels: list[int]) -> torch.Tensor:
        # create list of one-hots and sum down the class axis
        one_hot_class_labels = one_hot(
            torch.tensor(class_labels),
            num_classes=self.n_classes,
        )
        return torch.sum(one_hot_class_labels, dim=0)


def _extract_articles(text: str, article_1_marker: str):
    start = text.find(article_1_marker)

    if start == -1:
        return None

    return text[start:]


def extract_articles(item: LazyRow, languages: list[str]) -> dict[str, dict[str, str]]:
    return {
        "text": {
            lang: _extract_articles(
                text=item["text"][lang],
                article_1_marker=ARTICLE_1_MARKERS[lang],
            )
            for lang in languages
        }
    }


class TranslationPreProcesser:
    """Prepares the data for the translation task"""

    def __init__(self, language_pair: dict[str, str]) -> None:
        self.source_language = language_pair["source"]
        self.target_language = language_pair["target"]

    def __call__(
        self, data_row: dict[str, dict[str, str]]
    ) -> dict[str, str | dict[str, str]]:
        """
        processes the row in the dataset

        Args:
            data_row: input row

        Returns:
            row : processed row with relevant items
        """
        source_text = data_row["text"][self.source_language]
        target_text = data_row["text"][self.target_language]
        return {
            "source_text": source_text,
            "target_text": target_text,
        }


def load_mutlieurlex_metadata(data_dir: str, level: int) -> dict[str, Any]:
    assert level in [1, 2, 3], "there are 3 levels of hierarchy: 1,2,3."
    with open(f"{data_dir}/MultiEURLEX/data/eurovoc_concepts.json") as concepts_file:
        class_concepts = json.loads(concepts_file.read())
        concepts_file.close()

    with open(
        f"{data_dir}/MultiEURLEX/data/eurovoc_descriptors.json"
    ) as descriptors_file:
        class_descriptors = json.loads(descriptors_file.read())
        descriptors_file.close()
    # format level for the class descriptor dictionary, add these to a list
    classes = class_concepts[f"level_{level}"]
    descriptors = []
    for class_id in classes:
        descriptors.append(class_descriptors[class_id])

    # define metadata
    return {
        "n_classes": len(classes),
        "class_labels": classes,
        "class_descriptors": descriptors,
    }


def load_multieurlex(
    data_dir: str,
    level: int,
    languages: list[str],
) -> tuple[DatasetDict, dict[str, Any]]:
    """
    load the multieurlex dataset

    Args:
        data_dir: root directory for the dataset class descriptors and concepts
        level: level of hierarchy/specicifity of the labels
        languages: a list of iso codes for languages to be used

    Returns:
        List of datasets and a dictionary with some metadata information
    """
    metadata = load_mutlieurlex_metadata(data_dir=data_dir, level=level)

    # load the dataset with huggingface API
    if isinstance(languages, list):
        if len(languages) == 0:
            msg = "languages list cannot be empty"
            raise Exception(msg)

        load_langs = languages[0] if len(languages) == 1 else "all_languages"

    dataset_dict = load_dataset(
        "multi_eurlex",
        load_langs,
        label_level=f"level_{level}",
        trust_remote_code=True,
    )

    dataset_dict = dataset_dict.map(
        extract_articles, fn_kwargs={"languages": languages}
    )

    # return datasets and metadata
    return dataset_dict, metadata


def load_multieurlex_for_translation(
    data_dir: str, level: int, lang_pair: dict[str, str]
) -> tuple[DatasetDict, dict[str, Any]]:
    langs = [lang_pair["source"], lang_pair["target"]]
    dataset_dict, meta_data = load_multieurlex(
        data_dir=data_dir, level=level, languages=langs
    )
    # instantiate the preprocessor
    preprocesser = TranslationPreProcesser(lang_pair)
    # preprocess each split
    return dataset_dict.map(preprocesser, remove_columns=["text"]), meta_data
