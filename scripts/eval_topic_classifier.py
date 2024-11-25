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


def main(
    ckpt_path: str,
    data_root: str,
    batch_size: int,
    eval_output_dir: str,
    report_to: str = "tensorboard",
    dataset_name: str = "validation",
) -> None:
    model = DistilBertForSequenceClassification.from_pretrained(ckpt_path)
    tokenizer = DistilBertTokenizer.from_pretrained(ckpt_path)

    dataset_dict, metadata = load_multieurlex(
        data_dir=data_root, level=1, languages=["en"]
    )
    id2label = dict(enumerate(metadata["class_labels"]))

    dataset = dataset_dict[dataset_name]

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "id2label": id2label},
        remove_columns=["text", "labels"],
    )

    model = model.eval()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=eval_output_dir,
        per_device_eval_batch_size=batch_size,
        report_to=report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=tokenized_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.evaluate()


if __name__ == "__main__":
    cli.CLI(main, as_positional=False)
