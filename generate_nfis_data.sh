#!/bin/bash


#SBATCH --time=12:00:00
#SBATCH --account=def-dmatthew
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
module load python/3.9
module load geos
module load proj
export NFIS_PATH='/home/gclyne/scratch'
export NUM_CORES=32
export PROJECT_PATH='/home/gclyne/projects/def-dmatthew/gclyne/thesis'
source ~/ENV/bin/activate



python -m preprocessing.generation_runner environment=compute_canada project=/home/gclyne/projects/def-dmatthew/gclyne/thesis data=/home/gclyne/scratch
