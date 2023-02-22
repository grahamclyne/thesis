import pandas as pd
from omegaconf import DictConfig
import hydra
from preprocessing.utils import readCoordinates,getRollingWindow
import numpy as np

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    # t2m lai_lv swvl1 ro lai_hv sp skt evabs stl1 tp total_lai
    era_data = pd.read_csv(f'{cfg.data}/era_data.csv',index_col=False)
    nfis_data = pd.read_csv(f'{cfg.data}/generated_data/nfis_tree_cover_data.csv')

    era_data = era_data.rename(columns={'# year':'year'})
    era_data['lat'] = round(era_data['lat'],6)
    nfis_data['lat'] = round(nfis_data['lat'],6)
    df_merged = pd.merge(era_data,nfis_data,on=['year','lat','lon'],
                                                how='inner')
    for col in ['water','snow_ice','rock_rubble','exposed_barren_land','bryoids','shrubs','wetland',
        'wetland-treed','herbs','coniferous','broadleaf','mixedwood']:
        df_merged[col] = df_merged[col] / (df_merged['total_pixels'] - df_merged['no_change']) * 100
    df_merged['treeFrac'] = df_merged['coniferous'] + df_merged['broadleaf'] + df_merged['mixedwood'] + df_merged['wetland-treed']
    df_merged['baresoilFrac'] = df_merged['exposed_barren_land']
    df_merged['grassCropFrac'] = df_merged['bryoids'] + df_merged['herbs']
    df_merged['residualFrac'] = df_merged['snow_ice'] + df_merged['rock_rubble'] + df_merged['water']
    df_merged['wetlandFrac'] = df_merged['wetland']


    final_input = df_merged.rename(columns={'# tas_DJF':'tas_DJF','evabs':'evspsblsoi','stl1':'tsl','sp':'ps',
    'ro':'mrro','tp':'pr','swvl1':'mrsos'})
    #map names to input data names
    # final_input = final_input[np.count_nonzero(final_input.values, axis=1) > len(final_input.columns)-5]
    # final_input = final_input.fillna(0)
    final_input.to_csv(f'{cfg.data}/cleaned_observed_ann_input.csv',index=False)


    #generate transformer data

    
    inputs = cfg.model.input + ['year','lat','lon']
    final_input = final_input[inputs]
    managed_forest_coordinates = readCoordinates(f'{cfg.data}/managed_coordinates.csv',is_grid_file=False)
    seq_len = 30



    test_data = pd.DataFrame()
    for (lat,lon) in managed_forest_coordinates:
        lat = round(lat,6)
        lon = round(lon,7)
        grid_cell = final_input[np.logical_and(final_input['lat'] == lat,final_input['lon'] == lon)]
        rolling_window = getRollingWindow(grid_cell,seq_len,inputs)
        test_data = pd.concat([test_data,rolling_window],axis=0)
    test_data.to_csv(f'{cfg.data}/observed_timeseries{seq_len}_data.csv',index=False)

main()