import os

NFIS_PATH = os.getenv('NFIS_PATH')
ERA_PATH = os.getenv('PROJECT_PATH') + '/data/ERA'
SHAPEFILE_PATH = os.getenv('PROJECT_PATH') + '/data/shapefiles'
CESM_PATH = os.getenv('PROJECT_PATH') + '/data/CESM'
DATA_PATH = os.getenv('PROJECT_PATH') + '/data'
FORC_PATH = os.getenv('PROJECT_PATH') + '/data/FORC'
OUTPUT_PATH = DATA_PATH + '/output'
GENERATED_DATA = DATA_PATH + '/generated_data'