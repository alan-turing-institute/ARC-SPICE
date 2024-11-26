# finetune DistilBERT for topic classification over MultiEURLEX dataset


from jsonargparse import cli
from transformers import (
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)

from arc_spice.data.multieurlex_utils import load_multieurlex
from arc_spice.topic.utils import compute_metrics, preprocess_function


def main(data_root: str, training_args: dict):
    dataset, metadata = load_multieurlex(data_dir=data_root, level=1, languages=["en"])

    id2label = dict(enumerate(metadata["class_labels"]))
    label2id = {v: k for k, v in id2label.items()}

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert/distilbert-base-uncased"
    )

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "id2label": id2label},
        remove_columns=["text", "labels"],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(**training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    cli.CLI(main)
