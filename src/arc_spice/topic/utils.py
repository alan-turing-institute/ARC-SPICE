import torch
from sklearn.metrics import hamming_loss, zero_one_loss
from transformers import DistilBertTokenizer


def preprocess_function(
    examples, tokenizer: DistilBertTokenizer, id2label: dict[str, int]
):
    output = tokenizer(examples["text"], truncation=True, padding=True)
    labels = torch.zeros((len(examples["labels"]), len(id2label)), dtype=torch.float32)
    for i, label in enumerate(examples["labels"]):
        labels[i][label] = 1.0
    output["label"] = labels
    return output


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = (torch.sigmoid(torch.from_numpy(logits)) > 0.5).numpy().astype(int)
    zo_acc = 1 - zero_one_loss(labels, preds)
    ham_acc = 1 - hamming_loss(labels, preds)
    return {"zero_one_acc": zo_acc, "hamming_acc": ham_acc}
