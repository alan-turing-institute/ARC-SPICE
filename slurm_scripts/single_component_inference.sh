#!/bin/bash
#SBATCH --account vjgo8416-spice
#SBATCH --qos turing
#SBATCH --job-name SPICE_variational_RTC
#SBATCH --time 0-12:0:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --output /bask/projects/v/vjgo8416-spice/ARC-SPICE/slurm_scripts/slurm_logs/%j.out
#SBATCH --cpus-per-gpu 18

# Load required modules here
module purge
module load baskerville
module load bask-apps/live/live
module load Python/3.10.8-GCCcore-12.2.0


# change working directory
cd /bask/projects/v/vjgo8416-spice/ARC-SPICE/

source /bask/projects/v/vjgo8416-spice/ARC-SPICE/env/bin/activate

# change huggingface cache to be in project dir rather than user home
export HF_HOME="/bask/projects/v/vjgo8416-spice/hf_cache"

# TODO: script uses relative path to project home so must be run from home, fix
python scripts/single_component_inference.py
