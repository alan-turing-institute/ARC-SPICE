import json

import torch
from datasets import load_dataset
from datasets.formatting.formatting import LazyRow
from torch.nn.functional import one_hot
from torch.utils.data import Dataset

with open("data/MultiEURLEX/data/eurovoc_concepts.json", "r") as concepts_file:
    class_concepts = json.loads(concepts_file.read())
    concepts_file.close()

with open("data/MultiEURLEX/data/eurovoc_descriptors.json", "r") as descriptors_file:
    class_descriptors = json.loads(descriptors_file.read())
    descriptors_file.close()

ARTICLE_1_MARKERS = {"en": "\nArticle 1\n", "fr": "\nArticle premier\n"}


class MultiHot:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, class_labels):
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
    def __init__(self, language_pair):
        self.source_language = language_pair["source"]
        self.target_language = language_pair["target"]

    def __call__(self, data_row):
        source_text = data_row["text"][self.source_language]
        target_text = data_row["text"][self.target_language]
        labels = data_row["labels"]
        row = {
            "source_text": source_text,
            "target_text": target_text,
            "class_labels": labels,
        }
        return row


def load_multieurlex(level, lang_pair):

    assert level in [1, 2, 3], "there are 3 levels of hierarchy: 1,2,3."

    level = f"level_{level}"
    classes = class_concepts[level]
    descriptors = []
    for class_id in classes:
        descriptors.append(class_descriptors[class_id])

    data = load_dataset(
        "multi_eurlex",
        "all_languages",
        label_level=level,
        trust_remote_code=True,
    )

    meta_data = {
        "n_classes": len(classes),
        "class_labels": classes,
        "class_descriptors": descriptors,
    }
    preprocesser = PreProcesser(lang_pair)

    train_dataset = data["train"].map(preprocesser, remove_columns=["text"])
    extracted_train = train_dataset.map(
        extract_articles,
        fn_kwargs={"lang_pair": lang_pair},
    )
    test_dataset = data["test"].map(preprocesser, remove_columns=["text"])
    extracted_test = train_dataset.map(
        extract_articles,
        fn_kwargs={"lang_pair": lang_pair},
    )
    val_dataset = data["validation"].map(preprocesser, remove_columns=["text"])
    extracted_val = train_dataset.map(
        extract_articles,
        fn_kwargs={"lang_pair": lang_pair},
    )
    return [extracted_train, extracted_test, extracted_val], meta_data
