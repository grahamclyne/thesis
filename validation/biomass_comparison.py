import plotly.graph_objects as go
import xarray as xr
import pandas as pd
import other.config as config
import numpy as np
from functools import reduce
from sklearn.metrics import mean_squared_error,r2_score

cesm = pd.read_csv(f'{config.DATA_PATH}/cesm_data.csv') #t/ha
nfis_agb = pd.read_csv(f'{config.DATA_PATH}/nfis_agb.csv') 
predicted = pd.read_csv(f'{config.DATA_PATH}/forest_carbon_observed.csv')


cesm = cesm[cesm['# year'] == 2014]
predicted = predicted[predicted.year == 2015]

cesm['cesm_agb'] = cesm['cLeaf'] + cesm['cStem'] + cesm['cOther']
predicted['predicted_agb'] = predicted['cLeaf'] + predicted['cStem'] + predicted['cOther']
nfis_agb['agb'] = nfis_agb['agb'] / 10 #t/ha to kg/m2 is to divide by 10
predicted = predicted[predicted['predicted_agb'] > 0]
data_frames = [cesm,nfis_agb,predicted]
df_merged = reduce(lambda left,right: pd.merge(left,right,on=['lat','lon'],
                                            how='inner'), data_frames)
print(cesm,predicted)
# df_merged = df_merged.dropna(how='any')
df_merged = df_merged.fillna(0)
#rmse
cesm_rmse = sum((df_merged['cesm_agb'] - df_merged['agb'])**2)/len(df_merged)
pred_rmse = sum((df_merged['predicted_agb'] - df_merged['agb'])**2)/len(df_merged)
print('cesm_rmse: ',cesm_rmse)
print('prediced_rmse: ',pred_rmse)
print('cesm_r2: ',r2_score(df_merged['agb'],df_merged['cesm_agb']))
print('pred_r2:', r2_score(df_merged['agb'],df_merged['predicted_agb']))


#plot
comparison = df_merged[['predicted_agb','lat','lon']]
output = comparison.pivot(index='lat', columns='lon', values='predicted_agb')
fig = go.Figure(data=[go.Surface(z=output.values)])
fig.update_layout(title='NFIS reported Above-Ground Biomass', autosize=True)
fig.update_layout(
        scene = dict(
            yaxis = dict(
                tickmode = 'array',
                tickvals = list(range(0,len(output.index),3)),
                ticktext = [round(x,2) for x in output.index][::3],
                title='latitude'
                ),
            xaxis = dict(
                tickmode = 'array',
                tickvals = list(range(0,len(output.columns),5)),
                ticktext = output.columns[::5],
                title='longitude'
                ),
            zaxis = dict(
                tickvals = list(range(0,16,5)),
                ticktext = [0,5,15],
                title = 'kg/m^2'
            ),
            camera_eye= dict(x= 0.5, y= -1.7, z=1.),
            aspectratio = dict(x= 1, y= 1, z= 0.2)
        )
        # xaxis = dict(tickmode = 'array',ticktext = comparison['lat']),
        # yaxis = dict(tickmode = 'array',tickvals = list(range(0,len(comparison['lon']))),ticktext = comparison['lon'])
        )
fig.write_image('agb_biomass_predicted.png')