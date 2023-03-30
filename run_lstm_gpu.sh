#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=def-dmatthew
#SBATCH --gres=gpu:1 # request a GPU

#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
module load python/3.9
module load geos
module load proj


source ~/ENV/bin/activate
wandb login c1f678c655920120ec68e1dc542a9f5bab02dbfa
python -m train_lstm model=lstm environment=compute_canada project=/home/gclyne/projects/def-dmatthew/gclyne/thesis data=/home/gclyne/scratch
