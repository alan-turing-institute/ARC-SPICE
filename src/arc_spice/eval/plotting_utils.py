import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import check_consistent_length, column_or_1d
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.validation import _check_pos_label_consistency


def calibration_curve(
    y_true,
    y_prob,
    *,
    pos_label=None,
    n_bins=5,
    strategy="uniform",
    return_ece=False,
):
    """Compute true and predicted probabilities for a calibration curve.

    The method assumes the inputs come from a binary classifier, and
    discretize the [0, 1] interval into bins.

    Calibration curves may also be referred to as reliability diagrams.

    Read more in the :ref:`User Guide <calibration>`.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True targets.

    y_prob : array-like of shape (n_samples,)
        Probabilities of the positive class.

    pos_label : int, float, bool or str, default=None
        The label of the positive class.

        .. versionadded:: 1.1

    n_bins : int, default=5
        Number of bins to discretize the [0, 1] interval. A bigger number
        requires more data. Bins with no samples (i.e. without
        corresponding values in `y_prob`) will not be returned, thus the
        returned arrays may have less than `n_bins` values.

    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the widths of the bins.

        uniform
            The bins have identical widths.
        quantile
            The bins have the same number of samples and depend on `y_prob`.

    Returns
    -------
    prob_true : ndarray of shape (n_bins,) or smaller
        The proportion of samples whose class is the positive class, in each
        bin (fraction of positives).

    prob_pred : ndarray of shape (n_bins,) or smaller
        The mean predicted probability in each bin.

    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    See section 4 (Qualitative Analysis of Predictions).

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.calibration import calibration_curve
    >>> y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0])
    >>> prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=3)
    >>> prob_true
    array([0. , 0.5, 1. ])
    >>> prob_pred
    array([0.2  , 0.525, 0.85 ])
    """
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)
    pos_label = _check_pos_label_consistency(pos_label, y_true)

    if y_prob.min() < 0 or y_prob.max() > 1:
        err_msg = "y_prob has values outside [0, 1]."
        raise ValueError(err_msg)

    labels = np.unique(y_true)
    if len(labels) > 2:
        err_msg = f"Only binary classification is supported. Provided labels {labels}."
        raise ValueError(err_msg)
    y_true = y_true == pos_label

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        err_msg = (
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )
        raise ValueError(err_msg)

    binids = np.searchsorted(bins[1:-1], y_prob)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    if return_ece:
        ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))
        return prob_true, prob_pred, ece

    return prob_true, prob_pred


def classifier_ece(classification_vectors, meta_data, plotting=False):
    mlb = MultiLabelBinarizer(classes=meta_data["n_classes"])

    targets = mlb.fit_transform(classification_vectors["labels"])

    preds = np.stack(classification_vectors["mean_scores"])

    pipeline_ece_vals = [None] * meta_data["n_classes"]

    for class_label in range(meta_data["n_classes"]):
        class_preds = preds[:, class_label]
        class_targets = targets[:, class_label]

        prob_true, prob_pred, pipeline_ece_vals[class_label] = calibration_curve(
            y_true=class_targets,
            y_prob=class_preds,
            n_bins=10,
            pos_label=1,
            return_ece=True,
        )
        if plotting:
            plt.figure(figsize=(9, 8))
            plt.title(
                f"{(meta_data['class_descriptors'][class_label]['en']).capitalize()}  "
                f"ECE: {np.round(pipeline_ece_vals[class_label],3)}"
            )
            plt.bar(
                np.linspace(0, 1, len(prob_true)),
                prob_true,
                alpha=0.7,
                width=1 / len(prob_true),
                label="Class accuracy",
            )
            plt.bar(
                np.linspace(0, 1, len(prob_pred)),
                prob_pred,
                alpha=0.7,
                width=1 / len(prob_true),
                label="Predicted probability",
            )
            plt.plot([0, 1], [0, 1], "--", color="k", alpha=0.3)
            plt.legend()
            plt.xlabel("Confidence")
            plt.ylabel("Accuracy")
            plt.show()
            plt.show()
