import torch
from transformers import pipeline


def set_dropout(model, dropout_flag: bool):
    for _, param in model.named_modules():
        if isinstance(param, torch.nn.Dropout):
            # dropout on (True) -> want training mode train(True)
            # dropout off (False) -> eval mode train(False)
            param.train(dropout_flag)
    return model


def MCDropoutPipeline(task: str, model: str):
    pl = pipeline(
        task=task,
        model=model,
    )
    initial_model = pl.model
    pl.model = set_dropout(model=initial_model, dropout_flag=True)
    return pl


def test_dropout(pipe):
    model = pipe.model
    for name, param in model.named_modules():
        if isinstance(param, torch.nn.Dropout):
            print(name, param.training)
