#!/bin/bash
#
#SBATCH --job-name=SmallNet
#SBATCH --output=./out_files/%u_%j-%A.out
#
#SBATCH --ntasks=1
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --time=23:00:00
#
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hos002

source /gpfs/work/hos00/hos002/l1_setup.sh
srun python src/main.py -i /gpfs/work/hos00/hos002/datasets/imagenet -o ./results/ -j ${SLURM_JOB_ID} -a 1 -d 1001 -n 1 -c 40 -g False -s 400000 -b 32 -l 0.01 -p False -x False

