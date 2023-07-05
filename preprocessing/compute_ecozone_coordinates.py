import xarray as xr 
from preprocessing.utils import scaleLongitudes
import geopandas as gpd
import pandas as pd

#convert a list of tuples into a pandas dataframe
def convertToDF(list_of_tuples):
    df = pd.DataFrame(list_of_tuples, columns = ['lat','lon']) 
    return df

def getEcoZoneCoordinates():
    ecozones = gpd.read_file('data/shapefiles/ecozones.shp').to_crs('epsg:4326')
    netcdf_file = xr.open_dataset(f'data/CESM/treeFrac_Lmon_CESM2_historical_r11i1p1f1_gn_199901-201412.nc')
    netcdf_file = scaleLongitudes(netcdf_file)
    #to make sure the center of the cell is being considered for the clipping
    netcdf_file['lat'] = netcdf_file['lat'] + 0.5
    netcdf_file['lon'] = netcdf_file['lon'] + 0.75
    dissolved_ecozones = ecozones.dissolve(by='ZONE_NAME').reset_index()
    netcdf_file = netcdf_file['treeFrac']
    df = pd.DataFrame()
    for _,region in dissolved_ecozones.iterrows():
        gdf = gpd.GeoDataFrame(geometry=[region.geometry])
        netcdf_file.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        netcdf_file.rio.write_crs("epsg:4326", inplace=True)
        clipped = netcdf_file.rio.clip(gdf.geometry, ecozones.crs,drop=True)
        clipped['lat'] = clipped['lat'] - 0.5
        clipped['lon'] = clipped['lon'] - 0.75
        stacked = clipped.stack(x=['lat','lon'])
        val = stacked.isel(time=0)[stacked.isel(time=0).notnull()].coords['x'].values
        x = convertToDF(list(val))
        x['zone'] = region['ZONE_NAME']
        df= pd.concat([df,x])
    df['lat'] = df['lat'].round(7)
    return df

#SHOW STUDY AREA
import matplotlib.pyplot as plt
import geopandas 

ecozones = geopandas.read_file('data/shapefiles/ecozones.shp').to_crs('epsg:4326')
canada = geopandas.read_file('data/shapefiles/lpr_000b16a_e/lpr_000b16a_e.shp').to_crs('epsg:4326')
fig, ax = plt.subplots(figsize=(10,10))
canada.plot(ax=ax,color='white',edgecolor='black')
ax.legend(['Boreal Shield','Boreal Cordillera','Boreal Plain'],fontsize=30,loc='upper left')
ecozones['ZONE_NAME'] = ecozones['ZONE_NAME'].str.replace('Boreal PLain','Boreal Plain')
ecozones.where(ecozones['ZONE_NAME'].isin(['Boreal Shield','Boreal Cordillera','Boreal Plain'])).plot(column='ZONE_NAME',ax=ax,cmap='cividis',legend=True,legend_kwds={'fontsize':15})
# ax.set_title('Study Area',fontsize=40)
plt.savefig('figures/study_area.png',bbox_inches='tight')
