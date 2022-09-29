#!/bin/bash

#SBATCH --account=def-dmatthew
#SBATCH --time=01:00:00

export NFIS_PATH='~/scratch'
export NUM_CORES=32
export PROJECT_PATH='/home/gclyne/projects/def-dmatthew/gclyne/thesis'


module load python/3.9 
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
echo $PYTHONPATH
echo $PATH
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt
python /home/gclyne/projects/def-dmatthew/gclyne/thesis/other/generate_nfis_biomass_sums.py
rm -rf $ENVDIR
