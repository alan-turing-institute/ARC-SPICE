import json

from datasets import load_dataset

dataset = load_dataset(
    "multi_eurlex", "all_languages", label_level="level_3", trust_remote_code=True
)

for row_index in range(0, 10):
    row = dataset["train"][row_index]
    # print(json.dumps(row["celex_id"]))
    # print(json.dumps(row["text"]["en"], indent=2))
    print(json.dumps(row["labels"]))
