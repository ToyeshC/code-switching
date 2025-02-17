#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=extract_prompts
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=outputs/extract_prompts.out

module purge
module load 2024
source code-switch/bin/activate

python src/extract_prompts.py data/RTP-LX/RTP_LX_NL.json data/extracted_prompts/train_nl.txt