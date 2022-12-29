#generate sliding window timeseries of 30 years for each grid cell? 
import pandas as pd
import torch as T
from sklearn import preprocessing
import other.constants as constants
from other.utils import readCoordinates
from transformer_no_decoder import CMIPTimeSeriesDataset
from other.constants import MODEL_INPUT_VARIABLES,MODEL_TARGET_VARIABLES

cesm_data = pd.read_csv('/Users/gclyne/thesis/data/cesm_data.csv')
managed_forest_coordinates = readCoordinates('managed_coordinates.csv',is_grid_file=False)
input_variables = constants.MODEL_INPUT_VARIABLES
input_variable_tuple = tuple(constants.MODEL_INPUT_VARIABLES)
# output_variable_tuple = tuple(constants.MODEL_TARGET_VARIABLES)

min_max_scaler = preprocessing.MinMaxScaler([-0.5,0.5])
cesm_data.loc[:,input_variable_tuple] = min_max_scaler.fit_transform(cesm_data.loc[:,input_variable_tuple])
# cesm_data.loc[:,output_variable_tuple] = min_max_scaler.fit_transform(cesm_data.loc[:,output_variable_tuple])

inputs = MODEL_INPUT_VARIABLES + MODEL_TARGET_VARIABLES + ['# year','lat','lon']
cesm_data = cesm_data[inputs]
ds = CMIPTimeSeriesDataset(cesm_data,managed_forest_coordinates)
T.save(ds,'data.pt')