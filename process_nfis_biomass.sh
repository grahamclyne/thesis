#!/bin/bash

#SBATCH --account=def-dmatthew
#SBATCH --time=00:30:00
#SBATCH --mem=135G
module load python/3.9
module load geos
module load proj


source ~/ENV/bin/activate

python -m process_nfis_biomass
