#!/bin/bash
#SBATCH --account vjgo8416-spice
#SBATCH --qos turing
#SBATCH --job-name {{ job_name }}
#SBATCH --time {{ walltime }}
#SBATCH --nodes {{ node_number }}
#SBATCH --gpus {{ gpu_number }}
#SBATCH --output /bask/projects/v/vjgo8416-spice/ARC-SPICE/slurm_scripts/slurm_logs/{{ job_name }}-%j.out
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
export HF_HOME="{{ hf_cache_dir }}"

# TODO: script uses relative path to project home so must be run from home, fix
python {{ script_name }}
