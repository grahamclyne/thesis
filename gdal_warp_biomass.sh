#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --account=def-dmatthew
#SBATCH --mem=130G
module load gdal

gdalwarp -t_srs EPSG:4326 -r average /home/gclyne/scratch/CA_forest_total_biomass_2015.tif /home/gclyne/scratch/reprojected_4326_forest_total_biomass_2015.tif

