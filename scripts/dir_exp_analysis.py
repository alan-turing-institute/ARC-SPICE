"""
Run the full experiment analysis script for an entire directory.
"""

import os

from experiment_analysis import main as exp_main
from jsonargparse import CLI
from tqdm import tqdm


def get_paths(output_path: str) -> list[str]:
    """
    Get the paths of all the raw result .jsons from the output folder.

    Args:
        output_path: path to the output folder

    Returns:
        list of the output paths
    """
    all_paths = [
        os.path.join(dp, f)
        for dp, dn, fn in os.walk(os.path.expanduser(output_path))
        for f in fn
    ]
    return [path for path in all_paths if os.path.splitext(path)[-1] == ".json"]


def main(output_path: str):
    """
    Run the analysis for a whole series of experiments, providing just the top level
    directory.

    Args:
        output_path: outputs path
    """
    # get the paths for all the jsons
    result_paths = get_paths(output_path)

    # run analysis script
    for path in tqdm(result_paths):
        exp_main(path)


if __name__ == "__main__":
    CLI(main)
