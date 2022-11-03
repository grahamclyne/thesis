import pandas as pd
import other.config as config

data = pd.read_csv(f'{config.DATA_PATH}/cesm_data.csv')
data['grass_crop_shrub'] = data['cropFrac'] + data['grassFrac'] + data['shrubFrac']
data['exposed_land'] = data['residualFrac'] + data['baresoilFrac']