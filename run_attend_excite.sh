#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH -C pascal


cd ~/StableComics/AttendExcite/

nvidia-smi --list-gpus
nvidia-smi --query-gpu=memory.total --format=csv

python run_batch.py