import time
import csv
from preprocessing.generate_nfis_biomass import NFIS_Biomass
from preprocessing.generate_nfis_data import NFIS_Land_Cover
from preprocessing.utils import readCoordinates
import hydra
from omegaconf import DictConfig



#to run: python -m preprocessing.generation_runner --data_set nfis_land_cover --num_cores 2
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    
    start_time = time.time()

    coordinates = readCoordinates(f'{cfg.data}/{cfg.study_area}_coordinates.csv',is_grid_file=False)
    # ordered_latitudes = readCoordinates(f'{cfg.data}/grid_latitudes.csv',is_grid_file=True)
    # ordered_longitudes = readCoordinates(f'{cfg.data}/grid_longitudes.csv',is_grid_file=True)

    if(cfg.preprocess_data_set == 'nfis_biomass'):
        data_set = NFIS_Biomass(cfg)
    if(cfg.preprocess_data_set == 'nfis_land_cover'):
        data_set = NFIS_Land_Cover(cfg)

    observable_rows = open(data_set.output_file_path,'w')
    writer = csv.writer(observable_rows)
    writer.writerow(data_set.columns)
    num_cores = cfg.num_cores
    data_set.generate_data(coordinates,writer,num_cores)
    observable_rows.close()
   
    duration = time.time() - start_time
    print(f'Completed in {duration} seconds.')

if __name__ == '__main__': #need this for multiprocessing
    main()