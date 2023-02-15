#actual reported values

#cesm reported

#predicted

#hold out


#get reported values

import rasterio
import rioxarray
import xarray
import geopandas as gpd
from preprocessing.utils import scaleLongitudes


#get soil data to compare
soil_df = rioxarray.open_rasterio('/Users/gclyne/Downloads/McMaster_WWFCanada_soil_carbon1m_250m/McMaster_WWFCanada_soil_carbon1m_250m_kg-m2_version1.0.tif')
ref_df = xarray.open_dataset('/Users/gclyne/thesis/data/cesm/cSoilAbove1m_Emon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc')
ref_df = ref_df.rio.set_crs('epsg:4326')
soil_df = soil_df.rio.reproject_match(ref_df)
soil_df = soil_df.rename({'x':'lon','y':'lat'})
canada_mf_shapefile = gpd.read_file("data/shapefiles/MF.shp")
canada_mf_shapefile.to_crs(soil_df.rio.crs, inplace=True)
scaled_soil = scaleLongitudes(soil_df)
scaled_soil = scaled_soil.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
x = scaled_soil.rio.clip(canada_mf_shapefile.geometry.apply(lambda x: x.__geo_interface__), canada_mf_shapefile.crs, drop=True, invert=False, all_touched=False, from_disk=False)
ds_masked = x.where(x.data != x.rio.nodata)  
soil_pdf = ds_masked.sel(band=1).to_pandas()


soil_pdf = ds_masked.sel(band=1).to_dataframe(name='soil')
soil_pdf.reset_index(inplace=True)
soil_pdf.drop(columns=['band','spatial_ref'],inplace=True)
soil_pdf.dropna(inplace=True)

import pandas as pd
cesm_data = pd.read_csv('data/cesm_data_variant.csv')
cols = cesm_data.columns.append(pd.Index(['variant']))
cesm_data = cesm_data.reset_index()
cesm_data.columns = cols
cesm_data.rename(columns={'# year':'year'},inplace=True)
cesm_data = cesm_data.where(cesm_data['year'] > 1984).dropna()
cesm_data['lat'] = round(cesm_data['lat'],6)
cesm_data = cesm_data.groupby(['year','lat','lon']).mean().reset_index()
cesm_data = cesm_data[cesm_data['year'] == 2014]

import numpy as np
import matplotlib.pyplot as plt
obs_lstm = pd.read_csv('data/forest_carbon_observed_lstm.csv').dropna()
obs_lstm = obs_lstm[obs_lstm['year'] == 2015]

print(len(cesm_data),len(obs_lstm))
def plot3dCanada(data:pd.DataFrame,variable:str,title:str) -> None: 
    lat,lon = np.meshgrid(data['lat'].unique(),np.sort(data['lon'].unique()))
    grid = data.pivot_table(variable, 'lon', 'lat', fill_value=0).to_numpy()

    # Set up plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    surf = ax.plot_surface(lon,lat, grid,cmap='Greens',rstride=1,cstride=1,vmin=0,vmax=50)
    fig.colorbar(surf, shrink=0.5, aspect=4)

    ax.figure.set_size_inches(30,30)
    ax.view_init(50, -60)
    ax.set_ylabel('latitude',fontsize=30,labelpad=50)
    ax.set_xlabel('longitude',fontsize=30,labelpad=50)
    ax.set_title(title,fontsize=50,pad=50)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    ax.set_zlabel(variable,fontsize=30)
    plt.savefig(f'{title}.png',bbox_inches='tight')

plot3dCanada(cesm_data,'cSoilAbove1m','CESM2 Reported Soil Carbon')
plot3dCanada(soil_pdf,'soil','WWF Canada Reported Soil Carbon')
plot3dCanada(obs_lstm,'cSoilAbove1m','LSTM Predicted Soil Carbon (ERA)')

reported_sum = soil_pdf['soil'].sum()
cesm_sum = cesm_data['cSoilAbove1m'].sum()
lstm_predicted_with_observed_sum = obs_lstm['cSoilAbove1m'].sum()
cesm_rmse = np.sqrt(((soil_pdf['soil'] - cesm_data['cSoilAbove1m']) ** 2).mean())
lstm_rmse = np.sqrt(((soil_pdf['soil']  - obs_lstm['cSoilAbove1m']) ** 2).mean())

print(f'reported: {reported_sum}  cesm: {cesm_sum} lstm_predicted_with_observed: {lstm_predicted_with_observed_sum}')
print(f'cesm rmse: {cesm_rmse} lstm rmse: {lstm_rmse}')