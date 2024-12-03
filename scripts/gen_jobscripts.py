import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from jsonargparse import CLI

from arc_spice.utils import open_yaml_path

PROJECT_DIR = Path(__file__, "..", "..").resolve()


def main(experiment_config_path: str):
    """
    _summary_

    Args:
        experiment_config_path: _description_
    """
    experiment_name = experiment_config_path.split("/")[-1].split(".")[0]
    experiment_config = open_yaml_path(experiment_config_path)
    pipeline_conf_dir = (
        f"{PROJECT_DIR}/config/RTC_configs/{experiment_config['pipeline_config']}.yaml"
    )
    data_conf_dir = (
        f"{PROJECT_DIR}/config/data_configs/{experiment_config['data_config']}.yaml"
    )
    pipeline_config = open_yaml_path(pipeline_conf_dir)
    # Get jinja template
    environment = Environment(
        loader=FileSystemLoader(PROJECT_DIR / "src" / "arc_spice" / "config")
    )
    template = environment.get_template("jobscript_template.sh")
    # We don't want to overwrite results

    for index, seed in enumerate(experiment_config["seed"]):
        os.makedirs(
            f"slurm_scripts/experiments/{experiment_name}/run_{index}", exist_ok=False
        )
        for model in pipeline_config:
            model_script_dict: dict = experiment_config["bask"]
            model_script_dict.update(
                {
                    "script_name": (
                        "scripts/single_component_inference.py "
                        f"{pipeline_conf_dir} {data_conf_dir} {seed}"
                        f" {experiment_name} {model}"
                    ),
                    "job_name": f"{experiment_name}_{model}",
                    "seed": seed,
                }
            )
            model_train_script = template.render(model_script_dict)

            with open(
                f"slurm_scripts/experiments/{experiment_name}/run_{index}/{model}.sh",
                "w",
            ) as f:
                f.write(model_train_script)

        pipeline_script_dict: dict = experiment_config["bask"]
        pipeline_script_dict.update(
            {
                "script_name": (
                    "scripts/pipeline_inference.py "
                    f"{pipeline_conf_dir} {data_conf_dir} {seed}"
                    f" {experiment_name}"
                ),
                "job_name": f"{experiment_name}_full_pipeline",
                "seed": seed,
            }
        )
        pipeline_train_script = template.render(pipeline_script_dict)

        with open(
            f"slurm_scripts/experiments/{experiment_name}/run_{index}/full_pipeline.sh",
            "w",
        ) as f:
            f.write(pipeline_train_script)


if __name__ == "__main__":
    CLI(main)
