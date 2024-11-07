import torch
from transformers import Pipeline, pipeline


def set_dropout(model: torch.nn.Module, dropout_flag: bool) -> None:
    """
    Turn on or turn off dropout layers of a model.

    Args:
        model: pytorch model
        dropout_flag: dropout -> True/False
    """
    for _, param in model.named_modules():
        if isinstance(param, torch.nn.Dropout):
            # dropout on (True) -> want training mode train(True)
            # dropout off (False) -> eval mode train(False)
            param.train(dropout_flag)


def count_dropout(pipe: Pipeline, dropout_flag: bool) -> int:
    model = pipe.model
    dropout_count = 0
    for _, param in model.named_modules():
        if isinstance(param, torch.nn.Dropout):
            dropout_count += 1
            assert param.training == dropout_flag

    return dropout_count
