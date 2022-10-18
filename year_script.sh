#!/bin/bash

export PROJECT_PATH='/Users/gclyne/thesis'
export NUM_CORES=`sysctl -n hw.ncpu`
export NFIS_PATH='/Users/gclyne/thesis/data/NFIS'
for year in {2006..2020}
do
    python other/generate_gee_data.py --output_file_name modis_lai_data --variable_name lai --year $year  
done
python other/generate_gee_data.py --output_file_name elevation_data --variable_name elev --year 1984
