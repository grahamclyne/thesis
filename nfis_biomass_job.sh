#!/bin/bash

#SBATCH --account=def-dmatthew
#SBATCH --time=06:00:00
#SBATCH --mem=130G
export NFIS_PATH='/home/gclyne/scratch'
export NUM_CORES=32
export PROJECT_PATH='/home/gclyne/projects/def-dmatthew/gclyne/thesis'


module load python/3.9 
module load geos
ENVDIR=/tmp/$RANDOM
virtualenv --no-download $ENVDIR
source $ENVDIR/bin/activate
echo $PYTHONPATH
echo $PATH
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt
python /home/gclyne/projects/def-dmatthew/gclyne/thesis/other/generate_nfis_biomass_sums.py
deactivate
rm -rf $ENVDIR
