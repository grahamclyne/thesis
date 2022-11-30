import os 
import other.config as config
import pandas as pd

from other.constants import MODEL_INPUT_VARIABLES 

files = os.listdir()
data_frames = []

# t2m lai_lv swvl1 ro lai_hv sp skt evabs stl1 tp total_lai
era_data = pd.read_csv(f'{config.DATA_PATH}/era_data.csv',index_col=False)
nfis_data = pd.read_csv(f'{config.DATA_PATH}/generated_data/nfis_tree_cover_data.csv')

era_data = era_data.rename(columns={'# year':'year'})
era_data['lat'] = round(era_data['lat'],7)
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

print(MODEL_INPUT_VARIABLES)
print(final_input.columns)
# final_input = final_input.fillna(0)
final_input.to_csv(f'{config.DATA_PATH}/finalized_output.csv',index=False)