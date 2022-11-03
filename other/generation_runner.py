import time
import csv
from argparse import ArgumentParser
from other.generate_nfis_biomass import NFIS_Biomass
from other.generate_nfis_data import NFIS_Land_Cover
from other.generate_era_data import ERA_Dataset
from other.utils import readCoordinates

if __name__ == '__main__':
    
    start_time = time.time()

    managed_forest_coordinates = readCoordinates('managed_coordinates.csv',is_grid_file=False)
    ordered_latitudes = readCoordinates('grid_latitudes.csv',is_grid_file=True)
    ordered_longitudes = readCoordinates('grid_longitudes.csv',is_grid_file=True)

    parser = ArgumentParser()
    parser.add_argument('--data_set', type=str)
    args = parser.parse_args()
    if(args.data_set == 'nfis_biomass'):
        data_set = NFIS_Biomass()
    # data_set = NFIS_Land_Cover()
    # data_set = ERA_Dataset()

    observable_rows = open(data_set.output_file_path,'w')
    writer = csv.writer(observable_rows)
    writer.writerow(data_set.columns)
    num_cores = 1
    data_set.generate_data(managed_forest_coordinates,ordered_latitudes,ordered_longitudes,writer,num_cores)
    observable_rows.close()
    duration = time.time() - start_time
    print(f'Completed in {duration} seconds.')
