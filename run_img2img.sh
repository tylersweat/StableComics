#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --nodes=1


export LD_LIBRARY_PATH=/home/tysweat0/.conda/envs/img2img/lib/python3.9/site-packages/nvidia/cublas/lib

cd ~/StableComics/img2img/

nvidia-smi --list-gpus
nvidia-smi --query-gpu=memory.total --format=csv

python run.py