#!/bin/bash
#SBATCH --time=16:00:00
#SBATCH --account=def-dmatthew
#SBATCH --mem=130G
module load gdal

for i in {1984..2020}
do
    gdalwarp -t_srs EPSG:4326 -s_srs EPSG:3978 /home/gclyne/scratch/CA_forest_VLCE2_$i.tif /home/gclyne/scratch/reprojected_4326_CA_forest_$i.tif
done
gdalwarp -t_srs EPSG:4326 -s_srs EPSG:3978 /home/gclyne/scratch/CA_Forest_Harvest_1985-2020.tif /home/gclyne/scratch/reprojected_4326_CA_Forest_Harvest_1985-2020.tif
gdalwarp -t_srs EPSG:4326 -s_srs EPSG:9122 /home/gclyne/scratch/CA_forest_total_biomass_2015.tif /home/gclyne/scratch/reprojected_4326_forest_total_biomass_2015.tif
gdalwarp -t_srs EPSG:4326 /Users/gclyne/thesis/data/Base_Cur_AGB_MgCha_500m.tif /Users/gclyne/thesis/data/reprojected_4326_Base_Cur_AGB_MgCha_500m.tif