#!/bin/bash
#SBATCH --account vjgo8416-spice
#SBATCH --qos turing
#SBATCH --job-name baskerville_pipeline_inference_test_translator
#SBATCH --time 0-24:0:0
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --output /bask/projects/v/vjgo8416-spice/ARC-SPICE/slurm_scripts/slurm_logs/baskerville_pipeline_inference_test_translator-%j.out
#SBATCH --array=0-0
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
python scripts/single_component_inference.py /bask/projects/v/vjgo8416-spice/ARC-SPICE/config/RTC_configs/roberta-mt5-zero-shot.yaml /bask/projects/v/vjgo8416-spice/ARC-SPICE/config/data_configs/l1_fr_to_en.yaml 42 baskerville_pipeline_inference_test translator