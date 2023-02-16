import pandas as pd
from preprocessing.utils import readCoordinates,getRollingWindow
from omegaconf import DictConfig
import hydra
import numpy as np

    
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    cesm_data = pd.read_csv(f'{cfg.data}/cesm_data_variant.csv')
    # if('variant' in cfg.files.raw_data):
    cols = cesm_data.columns.append(pd.Index(['variant']))
    cesm_data = cesm_data.reset_index()
    cesm_data.columns = cols
    inputs = cfg.model.input + cfg.model.output + ['# year','lat','lon']
    cesm_data = cesm_data[inputs]
    cesm_data = cesm_data[cesm_data['# year'] < 1980]
    hold_out = cesm_data[cesm_data['# year'] >= 1980]
    managed_forest_coordinates = readCoordinates(f'{cfg.data}/managed_coordinates.csv',is_grid_file=False)
    seq_len = cfg.model.params.seq_len

    test_data = pd.DataFrame()
    for (lat,lon) in managed_forest_coordinates:
        lat = round(lat,7)
        lon = round(lon,7)
        grid_cell = cesm_data[np.logical_and(cesm_data['lat'] == lat,cesm_data['lon'] == lon)]
        rolling_window = getRollingWindow(grid_cell,seq_len,inputs)
        test_data = pd.concat([test_data,rolling_window],axis=0)
    test_data.to_csv(f'{cfg.data}/timeseries_cesm_training_data_{cfg.model.params.seq_len}.csv',index=False)

    hold_out_data = pd.DataFrame()
    for (lat,lon) in managed_forest_coordinates:
        lat = round(lat,7)
        lon = round(lon,7)
        grid_cell = hold_out[np.logical_and(hold_out['lat'] == lat,hold_out['lon'] == lon)]
        rolling_window = getRollingWindow(grid_cell,seq_len,inputs)
        hold_out_data = pd.concat([hold_out_data,rolling_window],axis=0)
    hold_out_data.to_csv(f'{cfg.data}/timeseries_cesm_hold_out_data_{cfg.model.params.seq_len}.csv',index=False)
main()