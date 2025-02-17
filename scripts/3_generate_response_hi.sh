#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=3_generate_response_hi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=/home/tchakravorty/tchakravorty/code-switching/outputs/3_generate_response_hi.out

module purge
module load 2024
source code-switch/bin/activate

# python ezswitch/src/inference.py \
#     --src data/extracted_prompts/train_en_small.txt \
#     --tgt data/extracted_prompts/train_hi_small.txt \
#     --gold_align data/alignments/train.en-hi.align \
#     --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
#     --output data/output/hindi/full_llama3_1.csv


python ezswitch/src/inference.py \
    --src data/extracted_prompts/train_en_small.txt \
    --tgt data/extracted_prompts/train_hi_small.txt \
    --src_translated data/translate_api_outputs/train_en_small.txt \
    --tgt_translated data/translate_api_outputs/train_hi_small.txt \
    --gold_align data/alignments/en-hi_align_gold.txt \
    --silver_src_align data/alignments/en-hi_align_silver.txt \
    --silver_tgt_align output/train.translated_llama-en-hi.align \
    # --human_reference data/hinge/train_human_generated.pkl \
    --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --output data/output/hindi/full_llama3_1.csv