import rasterio
import rioxarray
import xarray
import geopandas as gpd
from preprocessing.utils import scaleLongitudes
from sklearn import metrics
from preprocessing.utils import getArea
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
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




pd.set_option('display.max_columns', None)

soil_df = rioxarray.open_rasterio('/Users/gclyne/thesis/McMaster_WWFCanada_soil_carbon1m_250m/McMaster_WWFCanada_soil_carbon1m_250m_kg-m2_version1.0.tif')
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
soil_pdf = soil_pdf.rename(columns={'soil':'cSoilAbove1m'})
soil_pdf.reset_index(inplace=True)
soil_pdf.drop(columns=['band','spatial_ref'],inplace=True)
soil_pdf.dropna(inplace=True)




cesm_data = pd.read_csv('data/cesm_data_variant.csv')
inferred = pd.read_csv('data/forest_carbon_observed_lstm.csv')
hold_out_raw = pd.read_csv('data/forest_carbon_cesm_lstm.csv')
nfis_data = pd.read_csv('data/nfis_agb.csv')

#prep cesm data
cesm_data = cesm_data.where(cesm_data['year'] > 1984).dropna()
cesm_data['lat'] = round(cesm_data['lat'],6)
cesm_data = cesm_data.groupby(['year','lat','lon']).mean().reset_index()
cesm_data = cesm_data[cesm_data['year'] == 2014]
cesm_data['area'] = cesm_data.apply(lambda x: getArea(x['lat'],x['lon']),axis=1)

#prep hold_out data 
hold_out_raw = hold_out_raw[hold_out_raw['year'] == 2014]
hold_out_raw['lat'] = round(hold_out_raw['lat'],6)
hold_out = hold_out_raw.groupby(['year','lat','lon']).mean().reset_index()

#prep nfis agb 
nfis_data['agb'] = nfis_data['agb'] / 10 #ha to m2
nfis_data['lat'] = round(nfis_data['lat'],6)


inferred = inferred[inferred['year'] == 2014]
# cesm_data = cesm_data.rename(columns={'cSoilAbove1m_x':'cSoilAbove1m'})
# cesm_data.drop(columns=['cSoilAbove1m_y'],inplace=True)
# print(soil_pdf.lat.unique(),cesm_data.lat.unique())
soil_pdf['lat'] = round(soil_pdf['lat'],6)


hold_out.set_index(['lat','lon'],inplace=True)
inferred.set_index(['lat','lon'],inplace=True)
cesm_data.set_index(['lat','lon'],inplace=True)
soil_pdf.set_index(['lat','lon'],inplace=True)
nfis_data.set_index(['lat','lon'],inplace=True)
hold_out = hold_out.add_suffix('_hold_out')
inferred = inferred.add_suffix('_inferred')
cesm_data = cesm_data.add_suffix('_cesm_data')
soil_pdf = soil_pdf.add_suffix('_soil_pdf')
nfis_data = nfis_data.add_suffix('_nfis_data')

validation_df = pd.concat([hold_out,inferred,cesm_data,soil_pdf,nfis_data],axis=1,join='inner')
# plot3dCanada(cesm_data,'cSoilAbove1m','CESM2 Reported Soil Carbon')
# plot3dCanada(soil_pdf,'soil','WWF Canada Reported Soil Carbon')
# plot3dCanada(inferred,'cSoilAbove1m','LSTM Predicted Soil Carbon (ERA)')
validation_df.reset_index(inplace=True)
# soil_pdf = pd.merge(soil_pdf,cesm_data,on=['lat','lon'],how='inner')
# inferred = pd.merge(inferred,cesm_data,on=['lat','lon'],how='inner')
# hold_out = pd.merge(hold_out,cesm_data,on=['lat','lon'],how='inner')

# cesm_data = pd.merge(cesm_data,inferred,on=['lat','lon'],how='inner')

#HOLD OUT CALC

# hold_out['area'] = hold_out.apply(lambda x: getArea(x['lat'],x['lon']),axis=1)
# hold_out['cSoilAbove1m_x'] = hold_out['cSoilAbove1m_x'] * hold_out['area'] / 1e9
# merged['cSoilAbove1m_y'] = merged['cSoilAbove1m_y'] * merged['area'] / 1e9
# merged['agb_x'] = merged['cStem_x'] + merged['cLeaf_x'] + merged['cOther_x']
# merged['agb_y'] = merged['cStem_y'] + merged['cLeaf_y'] + merged['cOther_y']

# merged['agb_x'] = merged['agb_x'] * merged['area'] / 1e9
# merged['agb_y'] = merged['agb_y'] * merged['area'] / 1e9
for suffix in ['_hold_out','_inferred','_cesm_data','_soil_pdf','_nfis_data']:
    if(suffix != '_soil_pdf'):
        if(suffix != '_nfis_data'):
            validation_df['agb'+suffix] = validation_df['cStem'+suffix] + validation_df['cLeaf'+suffix] + validation_df['cOther'+suffix]
        validation_df['agb'+suffix] = validation_df['agb'+suffix] * validation_df['area_cesm_data'] / 1e9
    if(suffix != '_nfis_data'):
        validation_df['cSoilAbove1m'+suffix] = validation_df['cSoilAbove1m'+suffix] * validation_df['area_cesm_data'] / 1e9

    

# print(len(soil_pdf),len(cesm_data),len(inferred),len(hold_out))
# print(f'reported: {reported_sum}  cesm: {cesm_sum} lstm_predicted_with_observed: {lstm_predicted_with_observed_sum}')
# print(f'cesm rmse: {cesm_rmse} lstm rmse: {lstm_rmse}')
for suffix in ['_hold_out','_inferred','_soil_pdf','_nfis_data']:
    if(suffix != '_soil_pdf'):
        print(f'{suffix} AGB r2,rmse,sum : ',metrics.r2_score(validation_df['agb_cesm_data'],validation_df['agb'+suffix]),
              math.sqrt(metrics.mean_squared_error(validation_df['agb_cesm_data'],validation_df['agb'+suffix])),validation_df['agb'+suffix].sum())
        # print(f'AGB rmse {suffix}: ',)))
        # print(f'AGB sum {suffix}: ',validation_df['agb'+suffix].sum())

    if(suffix != '_nfis_data'):
        print(f'{suffix} SOIL r2 rmse sum : ',metrics.r2_score(validation_df['cSoilAbove1m_cesm_data'],validation_df['cSoilAbove1m'+suffix]),
              math.sqrt(metrics.mean_squared_error(validation_df['cSoilAbove1m_cesm_data'],validation_df['cSoilAbove1m'+suffix])),validation_df['cSoilAbove1m'+suffix].sum())
        # print(f'SOIL rmse {suffix}: '))
        # print(f'SOIL sum {suffix}: ')

print('AGB sum cesm: ',validation_df['agb_cesm_data'].sum())
print('SOIL sum cesm: ',validation_df['cSoilAbove1m_cesm_data'].sum())

# print(validation_df['cSoilAbove1m_cesm_data'],validation_df['cSoilAbove1m_inferred'])
# print('r2 observed: ',metrics.r2_score(soil_pdf['soil'],cesm_data['cSoilAbove1m_x']))
# print('r2 inferred: ',metrics.r2_score(inferred['soil'],cesm_data['cSoilAbove1m_x']))
# print('r2 hold_out: ',metrics.r2_score(hold_out['cSoilAbove1m_x'],cesm_data['cSoilAbove1m_x']))
# print('mse observed: ',metrics.mean_squared_error(soil_pdf['soil'],cesm_data['cSoilAbove1m_x']))
# print('mse inferred: ',metrics.mean_squared_error(inferred['soil'],cesm_data['cSoilAbove1m_x']))
# print('mse hold_out: ',metrics.mean_squared_error(hold_out['cSoilAbove1m_x'],cesm_data['cSoilAbove1m_x']))