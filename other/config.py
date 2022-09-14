import os

NFIS_PATH = os.getenv('NFIS_PATH')
ERA_PATH = os.getenv('PROJECT_PATH') + '/data/ERA'
SHAPEFILE_PATH = os.getenv('PROJECT_PATH') + '/data/shapefiles'
CESM_PATH = os.getenv('PROJECT_PATH') + '/data/CESM'
DATA_PATH = os.getenv('PROJECT_PATH') + '/data'
NUM_CORES = int(os.getenv('NUM_CORES'))
