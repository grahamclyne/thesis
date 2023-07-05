#!/bin/bash 
#SBATCH --time=4:00:00
#SBATCH --account=def-dmatthew
#SBATCH --mem=130G
module load gdal

gdalwarp -t_srs EPSG:4326 -s_srs EPSG:3978 -r average /home/gclyne/scratch/CA_Forest_Harvest_1985-2020.tif /home/gclyne/scratch/reprojected_4326_CA_Forest_Harvest_1985-2020.tif


