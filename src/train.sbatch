#!/bin/bash
#SBATCH --job-name=train_transformer
#SBATCH --cpus-per-task=4
#SBATCH --partition=week
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:4
#SBATCH --time=23:0:0
#SBATCH --output=jobs/transformer.%J.out
#SBATCH --error=jobs/train_transf.err

singularity exec --nv /common/singularityImages/DL23.sif python train_transformer.py 

echo DONE!