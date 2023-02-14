import time
import csv
from argparse import ArgumentParser
from preprocessing.generate_nfis_biomass import NFIS_Biomass
from preprocessing.utils import readCoordinates

if __name__ == '__main__':
    
    start_time = time.time()

    managed_forest_coordinates = readCoordinates(f'{cfg.path.data}/managed_coordinates.csv',is_grid_file=False)
    ordered_latitudes = readCoordinates('grid_latitudes.csv',is_grid_file=True)
    ordered_longitudes = readCoordinates('grid_longitudes.csv',is_grid_file=True)

    parser = ArgumentParser()
    parser.add_argument('--data_set', type=str)
    args = parser.parse_args()
    if(args.data_set == 'nfis_biomass'):
        data_set = NFIS_Biomass()
        observable_rows = open(data_set.output_file_path,'w')
        writer = csv.writer(observable_rows)
        writer.writerow(data_set.columns)
        num_cores = 5
        data_set.generate_data(managed_forest_coordinates,ordered_latitudes,ordered_longitudes,writer,num_cores)
        observable_rows.close()
    duration = time.time() - start_time
    print(f'Completed in {duration} seconds.')
