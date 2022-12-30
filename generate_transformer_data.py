import pandas as pd
from other.utils import readCoordinates
from omegaconf import DictConfig
import hydra
import numpy as np


def getRollingWindow(dataframe:pd.DataFrame,seq_len:int,cfg:DictConfig) -> pd.DataFrame:
    dataframe = dataframe[cfg.model.input + cfg.model.output + ['# year','lat','lon']]
    windows = pd.DataFrame()
    for window in dataframe.rolling(window=seq_len):
        if(len(window) == seq_len):
            windows = pd.concat([windows,window])
    return windows

    
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cesm_data = pd.read_csv(f'{cfg.path.data}/cesm_data.csv')
    inputs = cfg.model.input + cfg.model.output + ['# year','lat','lon']
    cesm_data = cesm_data[inputs]

    managed_forest_coordinates = readCoordinates('managed_coordinates.csv',is_grid_file=False)

    seq_len = 30



    test_data = pd.DataFrame()
    for (lat,lon) in managed_forest_coordinates:
        lat = round(lat,7)
        lon = round(lon,7)
        grid_cell = cesm_data[np.logical_and(cesm_data['lat'] == lat,cesm_data['lon'] == lon)]
        rolling_window = getRollingWindow(grid_cell,seq_len,cfg)
        test_data = pd.concat([test_data,rolling_window],axis=0)
    test_data.to_csv(f'{cfg.path.data}/cesm_transformer_data.csv',index=False)

main()