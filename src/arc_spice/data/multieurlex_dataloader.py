import json

from datasets import load_dataset

with open("data/MultiEURLEX/data/eurovoc_concepts.json") as concepts_file:
    class_concepts = json.loads(concepts_file)
    concepts_file.close()

with open("data/MultiEURLEX/data/eurovoc_descriptors.json") as descriptors_file:
    class_descriptors = json.loads(descriptors_file)
    descriptors_file.close()


def load_data(level):

    assert level in [1, 2, 3], "there are 3 levels of hierarchy: 1,2,3."

    level = f"level_{level}"
    classes = class_concepts[level]
    class_descriptors = []
    for class_id in classes:
        class_descriptors.append(class_descriptors[class_id])

    data = load_dataset(
        "multi_eurlex",
        "all_languages",
        label_level=f"level_{level}",
        trust_remote_code=True,
    )
    meta_data = {
        "n_classes": len(classes),
        "class_labels": classes,
        "class_descriptors": class_descriptors,
    }

    return data, meta_data
