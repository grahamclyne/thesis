import pandas as pd
import preprocessing.config as config 
from functools import reduce
import os 
from pickle import load
import torch as T
from ann.ann_model import Net,CMIPDataset
from preprocessing.constants import MODEL_TARGET_VARIABLES,MODEL_INPUT_VARIABLES

final_input = pd.read_csv(f'{config.DATA_PATH}/finalized_output.csv')
final_input = final_input.dropna(how='any')
scaler = load(open('/Users/gclyne/thesis/data/scaler.pkl', 'rb'))
data_to_estimate = scaler.transform(final_input[MODEL_INPUT_VARIABLES])
print(final_input[MODEL_INPUT_VARIABLES])
model = Net(len(MODEL_INPUT_VARIABLES),len(MODEL_TARGET_VARIABLES))

model.load_state_dict(T.load('/Users/gclyne/thesis/data/trained_net'))
ds = CMIPDataset(data_to_estimate,num_of_inputs=len(MODEL_INPUT_VARIABLES),num_of_targets=len(MODEL_TARGET_VARIABLES))
batch_size = 500
train_ldr = T.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=False)
results = []
for X in train_ldr:
    X = X[0]
    y = model(T.tensor(X.float()))
    results.extend(y.detach().numpy())
    
results_df = pd.DataFrame(results)
results_df['year'] = final_input['year']
results_df['lat'] = final_input['lat']
results_df['lon'] = final_input['lon']
print(results_df)
# output = results_df.groupby('year').mean()
results_df.columns = MODEL_TARGET_VARIABLES + ['year','lat','lon']
print(results_df)
results_df.to_csv(f'{config.DATA_PATH}/forest_carbon_observed.csv')
