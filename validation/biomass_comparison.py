import plotly.graph_objects as go
import xarray as xr
import pandas as pd
import other.config as config
import numpy as np
from functools import reduce

leaf = xr.open_dataset('/Users/gclyne/Downloads/cLeaf_Lmon_CESM2_historical_r11i1p1f1_gn_200001-201412.nc')
root = xr.open_dataset('/Users/gclyne/Downloads/cRoot_Lmon_CESM2_historical_r11i1p1f1_gn_200001-201412.nc')
veg = xr.open_dataset('/Users/gclyne/Downloads/cVeg_Lmon_CESM2_historical_r11i1p1f1_gn_200001-201412.nc')
stem = xr.open_dataset('/Users/gclyne/Downloads/cStem_Emon_CESM2_historical_r11i1p1f1_gn_200001-201412.nc')
other = xr.open_dataset('/Users/gclyne/Downloads/cOther_Emon_CESM2_historical_r11i1p1f1_gn_200001-201412.nc')

nfis_agb = pd.read_csv(f'{config.DATA_PATH}/nfis_agb.csv')

#t/ha
df_leaf = leaf.groupby('time.year').mean().sel(year=2014).to_dataframe().reset_index().drop(columns=['hist_interval','lat_bnds','lon_bnds','year']).groupby(['lat','lon']).mean().reset_index()
df_other = other.groupby('time.year').mean().sel(year=2014).to_dataframe().reset_index().drop(columns=['hist_interval','lat_bnds','lon_bnds','year']).groupby(['lat','lon']).mean().reset_index()
df_stem = stem.groupby('time.year').mean().sel(year=2014).to_dataframe().reset_index().drop(columns=['hist_interval','lat_bnds','lon_bnds','year']).groupby(['lat','lon']).mean().reset_index()
df_stem

df_leaf['lon'] = np.where(df_leaf['lon']> 180,df_leaf['lon'] - 360, df_leaf['lon']) 
df_other['lon'] = np.where(df_other['lon']> 180,df_other['lon'] - 360, df_other['lon']) 
df_stem['lon'] = np.where(df_stem['lon']> 180,df_stem['lon'] - 360, df_stem['lon']) 

df_leaf['lat'] = round(df_leaf['lat'],7)
df_other['lat'] = round(df_other['lat'],7)
df_stem['lat'] = round(df_stem['lat'],7)

data_frames = [df_leaf,df_other,df_stem,nfis_agb]
df_merged = reduce(lambda left,right: pd.merge(left,right,on=['lat','lon'],
                                            how='inner'), data_frames)


df_merged['cesm_agb'] = df_merged['cLeaf'] + df_merged['cOther'] + df_merged['cStem']
df_merged['agb'] = df_merged['agb'] / 10

#rmse
sum((df_merged['cesm_agb'] - df_merged['agb'])**2)/len(df_merged)


#plot
agb_df = pd.read_csv('/Users/gclyne/thesis/data/nfis_agb.csv')
comparison = pd.read_csv('/Users/gclyne/thesis/biomass_comparison.csv')
agb_df['agb'] = agb_df['agb'] / 10 #t/ha to kg/m2 is to divide by 10
comparison = comparison[['agb','lat','lon']]
output = comparison.pivot(index='lat', columns='lon', values='agb')
output = output.fillna(0)
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
fig.write_image('agb_biomass_nfis.png')