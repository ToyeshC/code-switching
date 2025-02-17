#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=4_compile_hindi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=02:00:00
#SBATCH --output=/home/tchakravorty/tchakravorty/code-switching/outputs/4_compile_hindi.out

module purge
module load 2024
source code-switch/bin/activate

python ezswitch/src/compile.py \
    --directory data/output/hindi \
    --output data/output/compile_hindi.csv