import pandas as pd
import other.config as config 
from functools import reduce
import os 
from pickle import load
import torch as T
from model import Net,CMIPDataset
from other.constants import MODEL_TARGET_VARIABLES,MODEL_INPUT_VARIABLES

final_input = pd.read_csv(f'{config.DATA_PATH}/finalized_output.csv')
print(final_input.columns)
scaler = load(open('/Users/gclyne/thesis/data/scaler.pkl', 'rb'))
data_to_estimate = scaler.transform(final_input[MODEL_INPUT_VARIABLES])

model = Net(len(MODEL_INPUT_VARIABLES),len(MODEL_TARGET_VARIABLES))

model.load_state_dict(T.load('/Users/gclyne/thesis/data/trained_net'))
ds = CMIPDataset(data_to_estimate,num_of_inputs=len(MODEL_INPUT_VARIABLES),num_of_targets=len(MODEL_TARGET_VARIABLES))

train_ldr = T.utils.data.DataLoader(ds,batch_size=1,shuffle=False)
results = []
for X in train_ldr:
    X = X[0]
    y = model(T.tensor(X.float()))
    results.append(y.detach().numpy()[0])
    
print(results)
years = [x for x in range(1984,2020,1)]
results_df = pd.DataFrame(results)
results_df['year'] = final_input['year']
results_df['lat'] = final_input['lat']
results_df['lon'] = final_input['lon']
# output = results_df.groupby('year').mean()
results_df.columns = MODEL_TARGET_VARIABLES + ['year','lat','lon']
results_df.to_csv("forest_carbon_observed.csv")

#tas matches w 2m temp, although there are several outliers
#lai is different although matches okay
#surface temp different by about 10 Kelvin
#precipitation way different
#bare soil observed 3%, cesm aroun 1.5%
#rsds should be rss
#surface pressure correct, off by about 3% 
#shrub/grass observed value almost 20% less
