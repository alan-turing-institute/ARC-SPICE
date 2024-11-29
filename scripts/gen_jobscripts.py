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
    experiment_config = open_yaml_path(experiment_config_path)
    pipeline_conf_dir = (
        f"{PROJECT_DIR}/config/RTC_configs/{experiment_config['pipeline_config']}.yaml"
    )
    data_conf_dir = (
        f"{PROJECT_DIR}/config/data_configs/{experiment_config['data_config']}.yaml"
    )
    pipeline_config = open_yaml_path(pipeline_conf_dir)
    # Get jinja template
    print(PROJECT_DIR / "src" / "arc_spice" / "configs")
    environment = Environment(
        loader=FileSystemLoader(PROJECT_DIR / "src" / "arc_spice" / "config")
    )
    template = environment.get_template("jobscript_template.sh")
    for model in pipeline_config:
        script_dict: dict = experiment_config["bask"]
        script_dict.update(
            {
                "script_name": (
                    "single_component_inference.py "
                    f"{pipeline_conf_dir} {data_conf_dir} {model}"
                ),
                "array_number": 0,
                "job_name": "test",
            }
        )
        train_script = template.render(script_dict)

        with open(f"temp/{model}_test.sh", "w") as f:
            f.write(train_script)


if __name__ == "__main__":
    CLI(main)
