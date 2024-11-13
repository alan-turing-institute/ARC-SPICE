from functools import partial

import torch
from transformers import Pipeline


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
    """
    counts the number of dropout layers that are in the configuration that is passed

    Args:
        pipe: pipeline over which to check dropout layers
        dropout_flag: the configuration in which they should be

    Returns:
        dropout_count: The number of layers in the correct configuration
    """
    model = pipe.model
    dropout_count = 0
    for _, param in model.named_modules():
        if isinstance(param, torch.nn.Dropout):
            dropout_count += 1
            assert param.training == dropout_flag

    return dropout_count


# original dropout function
dropout_orig_fn = torch.nn.functional.dropout


def dropout_w_training_override(
    input: torch.Tensor,
    p: float = 0.5,
    training: bool = True,
    inplace: bool = False,
    training_override: bool | None = None,
) -> torch.Tensor:
    """
    Overrides the dropout function to turn it on/off appropriately

    Args:
        ### Dropout function arguments
        input: input tensor
        p: dropout probability. Defaults to 0.5.
        training: training flag. Defaults to True.
        inplace: inplace flag. Defaults to False.
        ### Additional argument
        training_override: Overwrites the training argument to this value.
        Defaults to None.

    Returns:
        Dropout function with override on the training parameter
    """
    if training_override:
        training = training_override

    return dropout_orig_fn(input=input, p=p, training=training, inplace=inplace)


dropout_on = partial(dropout_w_training_override, training_override=True)
dropout_off = partial(dropout_w_training_override, training_override=False)
