import json

import torch
from datasets import load_dataset
from torch.nn.functional import one_hot
from torch.utils.data import Dataset

with open("data/MultiEURLEX/data/eurovoc_concepts.json", "r") as concepts_file:
    class_concepts = json.loads(concepts_file.read())
    concepts_file.close()

with open("data/MultiEURLEX/data/eurovoc_descriptors.json", "r") as descriptors_file:
    class_descriptors = json.loads(descriptors_file.read())
    descriptors_file.close()


class MultiEURLEXDataset(Dataset):

    def __init__(
        self,
        data,
        language_pair: dict[str:str, str:str],
        split,
    ) -> None:
        self.data = data[split]
        self.language_pair = language_pair

    def preprocess(self, data_row):
        source_text = data_row["text"][self.language_pair["source"]]
        target_text = data_row["text"][self.language_pair["target"]]
        labels = data_row["labels"]
        row = {
            "source_text": source_text,
            "target_text": target_text,
            "class_labels": labels,
        }
        return row

    def __getitem__(self, index):
        return self.preprocess(self.data[index])

    def __len__(self):
        return len(self.data)


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
    train_dataset = MultiEURLEXDataset(
        data=data, language_pair=lang_pair, split="train"
    )
    test_dataset = MultiEURLEXDataset(data=data, language_pair=lang_pair, split="train")
    val_dataset = MultiEURLEXDataset(
        data=data, language_pair=lang_pair, split="validation"
    )

    return [train_dataset, test_dataset, val_dataset], meta_data


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
