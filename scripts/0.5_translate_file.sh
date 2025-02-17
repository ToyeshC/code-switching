#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=0.5_translate_file
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=outputs/0.5_translate_file.out

module purge
module load 2024
source code-switch/bin/activate

python src/translate_file.py \
    --input data/extracted_prompts/train_hi_small.txt \
    --target en \
    --output data/translate_api_outputs/train_en_small.txt

python src/translate_file.py \
    --input data/extracted_prompts/train_en.txt \
    --target hi \
    --output data/translate_api_outputs/train_hi.txt

python src/translate_file.py \
    --input data/extracted_prompts/train_hi.txt \
    --target en \
    --output data/translate_api_outputs/train_en.txt
