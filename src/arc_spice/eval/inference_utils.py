import torch
from comet.models import CometModel

from arc_spice.data.multieurlex_utils import MultiHot
from arc_spice.eval.classification_error import hamming_accuracy


class ResultsGetter:
    def __init__(self, n_classes):
        self.multi_hot = MultiHot(n_classes)

    def get_results(
        self,
        clean_output: dict[str, dict],
        var_output: dict[str, dict],
        test_row: dict[str, list[int]],
        comet_model: CometModel,
        results_dict: dict[str, dict[str, list[float]]],
    ):
        # ### RECOGNITION ###
        # TODO: add this into results_getter

        # ### TRANSLATION ###
        source_text = test_row["target_text"]
        target_text = test_row["target_text"]
        clean_translation = clean_output["translation"]["full_output"]

        # load error model
        comet_inp = [
            {
                "src": source_text,
                "mt": clean_translation,
                "ref": target_text,
            }
        ]
        # comet doesn't work on MPS
        comet_device = "cuda" if torch.cuda.is_available() else "cpu"
        comet_output = comet_model.predict(
            comet_inp, batch_size=8, accelerator=comet_device
        )
        comet_scores = comet_output["scores"]
        results_dict["translation"]["comet_score"].append(comet_scores[0])
        results_dict["translation"]["weighted_semantic_density"].append(
            var_output["translation"]["weighted_semantic_density"]
        )

        # ### CLASSIFICATION ###
        mean_scores = var_output["classification"]["mean_scores"]
        preds = torch.round(mean_scores)
        hamming_acc = hamming_accuracy(
            preds=preds, class_labels=self.multi_hot(test_row["labels"])
        )
        results_dict["classification"]["hamming_accuracy"].append(hamming_acc)
        results_dict["classification"]["mean_predicted_entropy"].append(
            torch.mean(var_output["classification"]["predicted_entropy"]).item()
        )

        return results_dict
