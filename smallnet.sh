#!/bin/bash
#
#SBATCH --job-name=SmallNet
#SBATCH --output=./out_files/%u_%j-%A.out
#SBATCH --error=./out_files/%u_%j-%A.err
#
#SBATCH --ntasks=1
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --time=01:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hos002

srun python ./src/main.py %j

# srun -p gpus --nodes=1 --gres=gpu --time=01:00:00 --cpu_bind=none --pty /bin/bash -i
