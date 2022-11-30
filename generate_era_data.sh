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

#!/bin/bash
era=("evaporation_from_bare_soil" "leaf_area_index_high_vegetation" "leaf_area_index_low_vegetation"  "runoff"  "skin_temperature" "soil_temperature_level_1"  "surface_pressure"  "total_precipitation" "volumetric_soil_water_layer_1")

for i in ${era[@]};
do
PYTHONPATH=/home/gclyne/projects/def-dmatthew/gclyne/thesis python -m other.generation_runner --data_set era --era_variable $i
done