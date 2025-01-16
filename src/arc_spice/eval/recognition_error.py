from arc_spice.utils import flatten


def recognition_mean_scores(all_scores):
    mean_scores = []
    for row_scores in all_scores:
        flat_scores = flatten(row_scores)
        mean_scores.append(sum(flat_scores) / len(flat_scores))

    return mean_scores
