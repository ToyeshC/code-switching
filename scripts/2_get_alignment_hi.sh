#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=2_get_alignment_hi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=/home/tchakravorty/tchakravorty/code-switching/outputs/2_get_alignment_hi.out

module purge
module load 2024
source code-switch/bin/activate

# module load Boost/1.85.0-GCC-13.3.0

# cd /home/tchakravorty/tchakravorty/code-switching/ezswitch

# # Add the binary path to PATH using the absolute path
# export PATH=$PATH:/home/tchakravorty/tchakravorty/code-switching/ezswitch/alignment/giza-py/.bin

#  --bin ezswitch/alignment/giza-py/.bin \

# # Generate Alignment Files from gold translations
python ezswitch/alignment/giza-py/giza.py \
 --source data/extracted_prompts/train_en_small.txt \
 --target data/extracted_prompts/train_hi_small.txt \
 --alignments data/alignments/en-hi_align_gold.txt

# # Generate Alignment Files from silver translations
 python ezswitch/alignment/giza-py/giza.py \
 --source data/extracted_prompts/train_en_small.txt \
 --target data/translate_api_outputs/train_hi_small.txt \
 --alignments data/alignments/en-hi_align_silver.txt