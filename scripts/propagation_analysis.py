"""
Run analysis on the different models for propagation.
"""

import json

from jsonargparse import CLI

from arc_spice.eval.analysis_utils import (
    collect_pipeline_dict,
    exp_analysis,
    test_train_split_res,
)
from arc_spice.eval.prop_models import (
    eval_lin_models,
    eval_mult_prop,
    fit_uncertainty_model,
)
from arc_spice.utils import open_json_path


def main(experiment_path: str):
    """Run analysis of a given pipeline experiment using the different propagation
    models

    Args:
        experiment_path: path experiment directory
    """
    model_keys = ["ocr", "translator", "classifier"]

    # collect and collate results
    pipeline_results = open_json_path(f"{experiment_path}/full_pipeline.json")
    pipe_results = collect_pipeline_dict(pipeline_results)

    # no model results, rename keys
    no_mod_res = exp_analysis(pipe_results, model_keys)
    no_mod_res["recognition"] = no_mod_res.pop("ocr")
    no_mod_res["translation"] = no_mod_res.pop("translator")
    no_mod_res["classification"] = no_mod_res.pop("classifier")

    # multplication model resuls
    multi_mod_res = eval_mult_prop(pipe_results)

    # fitted model results
    train_res, test_res = test_train_split_res(pipe_results)
    fitted_uq_models = fit_uncertainty_model(train_res)
    fit_mod_res = eval_lin_models(fitted_uq_models, test_res)

    # collate results
    out_res = {}
    for key, itm in fit_mod_res.items():
        out_res[key] = {
            "no_model": no_mod_res[key],
            "mult_model": multi_mod_res[key],
            "fitted_model": itm,
        }

    # save results
    with open(f"{experiment_path}/prop_model_analysis.json", "w") as save_file:
        json.dump(out_res, save_file, indent=2)


if __name__ == "__main__":
    CLI(main)
