import os

NFIS_PATH = '~/scratch/'
ERA_PATH = os.getenv('PROJECT_DIR') + 'data/ERA/'
SHAPEFILE_PATH = os.getenv('PROJECT_DIR') + 'data/shapefiles/'
CESM_PATH = os.getenv('PROJECT_DIR') + 'data/CESM/'
DATA_PATH = os.getenv('PROJECT_DIR') + 'data/'
NUM_CORES = int(os.getenv('NUM_CORES'))
