#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=def-dmatthew
#SBATCH --mem=30G
module load gdal

`gdalwarp -t_srs EPSG:4326 -s_srs EPSG:3978 /Users/gclyne/thesis/data/NFIS/CA_Forest_Harvest_1985-2020/CA_Forest_Harvest_1985-2020.tif /Users/gclyne/thesis/reprojected_4326_CA_Forest_Harvest_1985-2020.tif`
for i in {1984..2020}
do
    gdalwarp -t_srs EPSG:4326 -s_srs EPSG:3978 /home/gclyne/scratch/CA_forest_VLCE2_$i.tif /Users/gclyne/thesis/reprojected_4326_CA_forest_$i.tif
done