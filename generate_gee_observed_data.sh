#!/bin/bash

export PROJECT_PATH='/Users/gclyne/thesis'
export NUM_CORES=`sysctl -n hw.ncpu`

python other/generate_gee_data.py --file_name modis_lai_data --variable_name lai  
python other/generate_gee_data.py --file_name elevation_data --variable_name elev 
