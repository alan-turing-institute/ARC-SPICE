import json
from typing import Union

import torch
from datasets import load_dataset
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
        one_hot_multi_class = torch.sum(one_hot_class_labels, dim=0)
        return one_hot_multi_class


def _extract_articles(text: str, article_1_marker: str):
    start = text.find(article_1_marker)

    if start == -1:
        return None

    return text[start:]


def extract_articles(item: LazyRow, lang_pair: dict[str:str]):
    lang_source = lang_pair["source"]
    lang_target = lang_pair["target"]
    return {
        "source_text": _extract_articles(
            text=item["source_text"],
            article_1_marker=ARTICLE_1_MARKERS[lang_source],
        ),
        "target_text": _extract_articles(
            text=item["target_text"],
            article_1_marker=ARTICLE_1_MARKERS[lang_target],
        ),
    }


class PreProcesser:
    """Function to preprocess the data, for the purposes of removing unused languages"""

    def __init__(self, language_pair: dict[str:str]) -> None:
        self.source_language = language_pair["source"]
        self.target_language = language_pair["target"]

    def __call__(
        self, data_row: dict[str : Union[str, list]]
    ) -> dict[str : Union[str, list]]:
        """
        processes the row in the dataset

        Args:
            data_row: input row

        Returns:
            row : processed row with relevant items
        """
        source_text = data_row["text"][self.source_language]
        target_text = data_row["text"][self.target_language]
        labels = data_row["labels"]
        row = {
            "source_text": source_text,
            "target_text": target_text,
            "class_labels": labels,
        }
        return row


def load_multieurlex(
    data_dir: str, level: int, lang_pair: dict[str:str]
) -> tuple[list, dict[str : Union[int, list]]]:
    """
    load the multieurlex dataset

    Args:
        data_dir: root directory for the dataset class descriptors and concepts
        level: level of hierarchy/specicifity of the labels
        lang_pair: dictionary specifying the language pair.

    Returns:
        List of datasets and a dictionary with some metadata information
    """
    assert level in [1, 2, 3], "there are 3 levels of hierarchy: 1,2,3."
    with open(
        f"{data_dir}/MultiEURLEX/data/eurovoc_concepts.json", "r"
    ) as concepts_file:
        class_concepts = json.loads(concepts_file.read())
        concepts_file.close()

    with open(
        f"{data_dir}/MultiEURLEX/data/eurovoc_descriptors.json", "r"
    ) as descriptors_file:
        class_descriptors = json.loads(descriptors_file.read())
        descriptors_file.close()
    # format level for the class descriptor dictionary, add these to a list
    level = f"level_{level}"
    classes = class_concepts[level]
    descriptors = []
    for class_id in classes:
        descriptors.append(class_descriptors[class_id])

    # load the dataset with huggingface API
    data = load_dataset(
        "multi_eurlex",
        "all_languages",
        label_level=level,
        trust_remote_code=True,
    )
    # define metadata
    meta_data = {
        "n_classes": len(classes),
        "class_labels": classes,
        "class_descriptors": descriptors,
    }
    # instantiate the preprocessor
    preprocesser = PreProcesser(lang_pair)
    # preprocess each split
    dataset = data.map(preprocesser, remove_columns=["text"])
    extracted_dataset = dataset.map(
        extract_articles,
        fn_kwargs={"lang_pair": lang_pair},
    )
    # return datasets and metadata
    return [
        extracted_dataset["train"],
        extracted_dataset["test"],
        extracted_dataset["validation"],
    ], meta_data
