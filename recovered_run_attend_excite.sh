#!/bin/bash

#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH -C pascal


cd ~/Attend-and-Excite/

nvidia-smi --list-gpus
nvidia-smi --query-gpu=memory.total --format=csv

python3 run_batch.py
