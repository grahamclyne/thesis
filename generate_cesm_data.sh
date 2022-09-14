#!/bin/bash

export PROJECT_PATH='/Users/gclyne/thesis'
export NUM_CORES=`sysctl -n hw.ncpu`
 
python other/generate_cesm_data.py
