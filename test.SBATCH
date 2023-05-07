#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:4
#SBATCH --job-name=torch
#SBATCH --output="%A\_%x.txt"
#SBATCH --account=ece_gy_9143-2023sp
#SBATCH --partition=n1c24m128-v100-4

echo "Processor: $(lscpu | grep 'Model name' | awk -F ':' '{print $2}' | xargs)"
echo "RAM: $(free -h | grep Mem: | awk '{print $4}')"
echo "GPU: $(nvidia-smi -q | grep 'Product Name')"

singularity exec --nv \
	    --overlay /scratch/sz3714/pytorch-example/my_pytorch.ext3:ro \
	    /share/apps/images/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python Q1.py"
