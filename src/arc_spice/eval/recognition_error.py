from arc_spice.eval.translation_error import length_normalised_metric


def mean_score(score_list):
    score_lens = []
    score_means = []
    for score in score_list:
        score_lens.append(len(score))
        score_means.append(sum(score) / len(score))

    return length_normalised_metric(score_lens, score_means)


def recognition_mean_scores(all_scores):
    return [mean_score(row_scores) for row_scores in all_scores]
