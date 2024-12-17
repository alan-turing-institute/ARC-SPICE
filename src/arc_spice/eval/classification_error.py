import math

import torch


def aggregate_score(probs: torch.Tensor) -> torch.Tensor:
    # average 'distance' from the predicted class
    preds = torch.round(probs).float()
    distance = torch.abs(preds - probs)
    return 1 - torch.mean(distance)


def MC_dropout_scores(
    variational_probs: list[float], epsilon: float = 1e-14
) -> dict[str, torch.Tensor]:
    # aggregate over the classes, performing MC Dropout on each class treating it
    # as a binary classification problem
    stacked_probs = torch.stack(
        [torch.tensor(out) for out in variational_probs], dim=-1
    )
    means = torch.mean(stacked_probs, dim=-1)

    pred_entropies = -1 * (
        means * torch.log(means + epsilon)
        + (1 - means) * torch.log((1 - means) + epsilon)
    )

    all_entropies = -1 * (
        stacked_probs * torch.log(stacked_probs + epsilon)
        + (1 - stacked_probs) * torch.log((1 - stacked_probs) + epsilon)
    )

    mutual_info = pred_entropies - torch.mean(all_entropies, dim=-1)

    stds = torch.std(stacked_probs, dim=-1)
    variances = torch.var(stacked_probs, dim=-1)

    return {
        "mean_outputs": means,
        "standard_deviations": stds,
        "variance": variances,
        "predicted_entropy": pred_entropies,
        "all_entropies": all_entropies,
        "mutual_information": mutual_info,
    }


def clean_entropy(clean_scores):
    epsilon = 1e-15
    stacked_scores = torch.stack(
        [torch.tensor(row_scores) for row_scores in clean_scores]
    )
    clean_entropies = (
        -1
        * (
            stacked_scores * torch.log(stacked_scores + epsilon)
            + (1 - stacked_scores) * torch.log((1 - stacked_scores) + epsilon)
        )
        / math.log(2)
    )
    return torch.mean(clean_entropies, dim=-1).tolist()
