#!/bin/bash


#SBATCH --time=12:00:00
#SBATCH --account=def-dmatthew
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
module load python/3.9
module load geos
module load proj
export NFIS_PATH='/home/gclyne/scratch'
export NUM_CORES=32
export PROJECT_PATH='/home/gclyne/projects/def-dmatthew/gclyne/thesis'
source ~/ENV/bin/activate
wandb login c1f678c655920120ec68e1dc542a9f5bab02dbfa
#python ${PROJECT_PATH}/other/generate_nfis_data.py
python ${PROJECT_PATH}/run_transformer.py path.project='/home/gclyne/projects/def-dmatthew/gclyne/thesis/' path.data='/home/gclyne/projects/def-dmatthew/gclyne/thesis/data'
