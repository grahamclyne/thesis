import pandas as pd
from preprocessing.utils import readCoordinates,getRollingWindow
from omegaconf import DictConfig
import hydra
import numpy as np

    
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    cesm_data = pd.read_csv(f'{cfg.path.data}/cesm_data_variant.csv')
    # if('variant' in cfg.files.raw_data):
    cols = cesm_data.columns.append(pd.Index(['variant']))
    cesm_data = cesm_data.reset_index()
    cesm_data.columns = cols
    inputs = cfg.model.input + cfg.model.output + ['# year','lat','lon']
    cesm_data = cesm_data[inputs]
    managed_forest_coordinates = readCoordinates(f'{cfg.path.data}/managed_coordinates.csv',is_grid_file=False)
    seq_len = 30
    test_data = pd.DataFrame()
    for (lat,lon) in managed_forest_coordinates:
        lat = round(lat,7)
        lon = round(lon,7)
        grid_cell = cesm_data[np.logical_and(cesm_data['lat'] == lat,cesm_data['lon'] == lon)]
        rolling_window = getRollingWindow(grid_cell,seq_len,inputs)
        test_data = pd.concat([test_data,rolling_window],axis=0)
    test_data.to_csv(f'{cfg.path.data}/{cfg.files.transformer_train_data}',index=False)

main()