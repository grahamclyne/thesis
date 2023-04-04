#!/bin/bash


#SBATCH --time=16:00:00
#SBATCH --account=def-dmatthew

#SBATCH --mem=30G
module load python/3.9
module load geos
module load proj


source ~/ENV/bin/activate
python -m test_preprocess