#!/bin/bash


#SBATCH --time=12:00:00
#SBATCH --account=def-dmatthew
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
module load python/3.9
module load geos
module load proj


source ~/ENV/bin/activate
wandb login c1f678c655920120ec68e1dc542a9f5bab02dbfa
python train_lstm.py +model=lstm +environment=compute_canada project=/home/gclyne/projects/def-dmatthew/gclyne/thesis data=/home/gclyne/scratch
