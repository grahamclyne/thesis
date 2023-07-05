import pandas as pd
from omegaconf import DictConfig
import hydra
from preprocessing.utils import readCoordinates,getRollingWindow
import numpy as np

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    era_data = pd.read_csv(f'{cfg.data}/ERA/era_data_{cfg.study_area}.csv',index_col=False)
    era_data = era_data.dropna()
    nfis_data = pd.read_csv(f'{cfg.data}/forest_df.csv')
    era_data = era_data.rename(columns={'# year':'year'})
    era_data['lat'] = round(era_data['lat'],6)
    nfis_data['lat'] = round(nfis_data['lat'],6)
    df_merged = pd.merge(era_data,nfis_data,on=['year','lat','lon'],
                                                how='inner')
    

    # for col in ['water','snow_ice','rock_rubble','exposed_barren_land','bryoids','shrubs','wetland',
    #     'wetland-treed','herbs','coniferous','broadleaf','mixedwood']:
    #     df_merged[col] = df_merged[col] / (df_merged['total_pixels'] - df_merged['no_change']) * 100
    # df_merged['treeFrac'] = df_merged['coniferous'] + df_merged['broadleaf'] + df_merged['mixedwood'] + df_merged['wetland-treed']
    # df_merged['baresoilFrac'] = df_merged['exposed_barren_land']
    # df_merged['grassCropFrac'] = df_merged['bryoids'] + df_merged['herbs']
    # df_merged['residualFrac'] = df_merged['snow_ice'] + df_merged['rock_rubble'] + df_merged['water']
    # df_merged['wetlandFrac'] = df_merged['wetland']
    df_merged['tree_cover'] = df_merged['tree_cover'] * 100

    
    final_input = df_merged.rename(columns={'# tas_DJF':'tas_DJF','evabs':'evspsblsoi','stl1':'tsl','sp':'ps',
    'ro':'mrro','tp':'pr','swvl1':'mrsos','tree_cover':'treeFrac'})
    
    #drop all zero columsn
    final_input = final_input[final_input['ps'] != 0]

    final_input.to_csv(f'{cfg.data}/cleaned_observed_ann_input.csv',index=False)


    #generate transformer data
    inputs = cfg.model.input + ['year','lat','lon']
    final_input = final_input[inputs]
    coordinates = readCoordinates(f'{cfg.data}/{cfg.study_area}_coordinates.csv',is_grid_file=False)
    seq_len = cfg.model.seq_len
    test_data = pd.DataFrame()
    for (lat,lon) in coordinates:
        lat = round(lat,6)
        lon = round(lon,7)
        grid_cell = final_input[np.logical_and(final_input['lat'] == lat,final_input['lon'] == lon)]
        rolling_window = getRollingWindow(grid_cell,seq_len,inputs)
        test_data = pd.concat([test_data,rolling_window],axis=0)
    test_data.to_csv(f'{cfg.data}/observed_timeseries{seq_len}_data.csv',index=False)

main()