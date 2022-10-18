import pandas as pd
import other.config as config 
from functools import reduce
import os 
from pickle import load
import torch as T
from draft_nn import Net,CMIPDataset

os.chdir(f'{config.DATA_PATH}/generated_observable_data')
files = os.listdir()
data_frames = []
for file in files:
    data_frames.append(pd.read_csv(file))

# data_frames = [temp,precip,nfis_tree_cover,surface_pressure,solar_radiation,high_lai,low_lai]
df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year','lat','lon'],
                                            how='outer'), data_frames)

df_merged['total_lai'] = df_merged['leaf_area_index_low_vegetation.nc'] + df_merged['leaf_area_index_high_vegetation.nc']
#('pr','tas','# lai','treeFrac','baresoilFrac','ps','grass_crop_shrub')
df_merged['total_precipitation.nc'] = df_merged['total_precipitation.nc'] / (24*60*60*30.437) *1000
years = df_merged['year']

final_input = df_merged[['total_precipitation.nc','2m_temperature.nc','total_lai','observed_tree_cover',
'observed_bare','surface_pressure.nc','observed_shrub_bryoid_herb']]


scaler = load(open('/Users/gclyne/thesis/data/scaler.pkl', 'rb'))
final_input.columns = ['pr','tas','# lai','treeFrac','baresoilFrac','ps','grass_crop_shrub']
final_input = scaler.transform(final_input)
#cesm precip = kg/m2/s
#era precip = m/month

#era -> cesm : 
# precip_adjusted = precip / (24*60*60*30.437) * 1000 #1000 here bc 1kg/m^2 is 1mm of thickness

model = Net(7)

model.load_state_dict(T.load('/Users/gclyne/thesis/data/trained_net'))
print(final_input[0])
dataset = CMIPDataset(final_input,7)
train_ldr = T.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
results = []
for X in train_ldr:
    X = X[0]
    y = model(T.tensor(X.float()))
    results.append(y.detach().numpy()[0])
    
print(results)
results_df = pd.DataFrame(results,years)
output = results_df.groupby('year').mean()
output.columns = ['cSoil','cCwd','cVeg','cLitter']
output.to_csv("forest_carbon_observed.csv")

#tas matches w 2m temp, although there are several outliers
#lai is different although matches okay
#surface temp different by about 10 Kelvin
#precipitation way different
#bare soil observed 3%, cesm aroun 1.5%
#rsds should be rss
#surface pressure correct, off by about 3% 
#shrub/grass observed value almost 20% less
