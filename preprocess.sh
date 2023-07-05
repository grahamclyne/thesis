#!/bin/bash


#SBATCH --time=3:00:00
#SBATCH --account=def-dmatthew

#SBATCH --mem=200G
module load python/3.9
module load geos
module load proj


source ~/ENV/bin/activate
python -m preprocessing.preprocess_nfis