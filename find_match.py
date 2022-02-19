import xarray as xr
import time
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Point
import logging
from datetime import datetime
import csv


def getClimateData(lat,lon,year,climate_data,variable):
    return climate_data[variable].sel(
        year=year,
        latitude=lat, 
        longitude=lon, method='nearest').mean().values  

def compareTreeCover():
    return 

def getERAData(lat,lon,year,era_data,variable):
    yearly_avg = era_data[variable].isel(
        year=year - 2000,
        latitude=np.logical_and(era_data.latitude >= lat,era_data.latitude < lat+0.9), 
        longitude=np.logical_and(era_data.longitude >= lon, era_data.longitude < lon+1.25)
        ).mean().values  
    return yearly_avg

def getElevation(lat,lon,elevation_data):
    lon = lon - 360 if lon > 180 else lon
    elevation = elevation_data.isel(y=np.logical_and(elevation_data.y>lat,elevation_data.y<lat+1),
        x=np.logical_and(elevation_data.x>lon,elevation_data.x<lon+1.25),band=elevation_data.band==1)['band_data'].mean().values
    return elevation

def getGPP(lat,lon,year,gpp_data):
    gpp = gpp_data['gpp'].isel(
        year=year - gpp_data['year'][0].values,
        lat=np.logical_and(gpp_data.lat >= lat,gpp_data.lat < lat+0.9), 
        lon=np.logical_and(gpp_data.lon >= lon, gpp_data.lon < lon+1.25)
        ).mean()
    return gpp

def getAGB(lat,lon,year,agb_data):
    gpp = gpp_data['agb'].isel(
        year=year - gpp_data['year'][0].values,
        lat=np.logical_and(gpp_data.lat >= lat,gpp_data.lat < lat+0.9), 
        lon=np.logical_and(gpp_data.lon >= lon, gpp_data.lon < lon+1.25)
        ).mean()    
    return agb
def getTreeCover(lat,lon,year,tree_cover_data):
    tree_cover = tree_cover_data['treeFrac'].isel(
        year=year - tree_cover_data['year'][0].values,
        lat=np.logical_and(tree_cover_data.lat >= lat,tree_cover_data.lat < lat+0.9), 
        lon=np.logical_and(tree_cover_data.lon >= lon, tree_cover_data.lon < lon+1.25)
        ).mean().values   
    return tree_cover
# x = datetime.strptime(str(year.values)[:19], '%Y-%m-%d %H:%M:%S')
#slicing to 19th place is to remove trailing .xxxx in time that strptime doesnt handle

def getObservedTreeCover(lat,lon,year,tree_cover_data):
    return 0 

def findSuitableCell(lat_range,lon_range,year,elevation_data,era_data,tree_cover_data,observed_data):
    for lat in lat_range:
        for lon in lon_range:
            elevation = getElevation(lat,lon,elevation_data)
            precipitation = getClimateData(lat,lon,year,era_data,'tp')
            t2m = getClimateData(lat,lon,year,era_data,'t2m')
            tree_cover = getTreeCover(lat,lon,year,tree_cover_data)
            if (compareValues([elevation,precipitation,t2m,tree_cover], observed_data)):
                return (lat,lon)
    return (0,0)
   

def compareValues(simulated_values, observed_values):
    print(simulated_values, observed_values)
    for index in range(len(simulated_values)):
        diff = abs(simulated_values[index] - observed_values[index])
        print(diff)
        if(diff/2 > observed_values[index]):
            return False
    return True

if __name__ == "__main__":
    start = time.time()
    logging.basicConfig(filename='/home/graham/code/thesis/logs/lookup_table_'+ str(datetime.now()) + ' .log', level=logging.DEBUG)
    output = open('output.csv', 'w')
    csv_writer = csv.writer(output)
    data_folder = '/home/graham/code/thesis/data/'
    # load zips with the source projection
    shapefilename = '/home/graham/code/thesis/boreal_reduced.shp'
    zips = gpd.read_file(shapefilename)
    # convert projection to familiar lat/lon
    # zips = zips.to_crs('epsg:4326')    
    tree_cover_data = xr.open_dataset(data_folder + 'treeFrac_Lmon_CESM2_land-hist_r1i1p1f1_gn_194901-201512.nc')
    gpp_data = xr.open_dataset(data_folder + 'gpp_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc')
    agb_data = xr.open_dataset(data_folder + 'cVeg_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc')
    tree_frac_file = data_folder + 'treeFrac_Lmon_CESM2_land-hist_r1i1p1f1_gn_194901-201512.nc'
    temperature_file = data_folder + 'tas_Amon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc'
    era_data = xr.open_dataset(data_folder + 'era_groupedby_time.nc').load() 
    temp_set = xr.open_dataset(temperature_file, decode_times=True, decode_cf=True  )
    temp_set = temp_set.groupby('time.year').mean('time')
    gpp_data = gpp_data.groupby('time.year').mean('time')
    tree_cover_data = tree_cover_data.groupby('time.year').mean('time')
    temp_set = temp_set.isel(lat=np.logical_and(temp_set.lat<70,temp_set.lat > 50))
    temp_set = temp_set.isel(lon=np.logical_and(temp_set.lon-360>-180,temp_set.lon-360 <-90))
    temp_set = temp_set.isel(year=temp_set.year>1999)
    elevation_data = xr.open_dataset(data_folder + 'a10g',engine='rasterio')
    tree_cover_2000 = rasterio.open(data_folder + 'Hansen_GFC-2020-v1.8_treecover2000_60N_130W.tif')
    tree_cover_loss_year = rasterio.open(data_folder + 'Hansen_GFC-2020-v1.8_lossyear_60N_130W.tif')
    #check if grid cell in boreal forest
    for year in temp_set['year'].values:    
        total_gpp = 0
        total_agb = 0
        for lat in temp_set['lat'].values:
            for lon in temp_set['lon'].values:
                #is this in boreal forest?
                lon_scaled = lon - 360 if lon > 180 else lon
                point = Point(lon_scaled,lat)
                if(not np.all(zips['geometry'].contains(point))):
                    continue
                elevation = getElevation(lat,lon,elevation_data)
                precipitation = getERAData(lat,lon,year,era_data,'tp')
                t2m = getERAData(lat,lon,year,era_data,'t2m')
                tree_cover = getTreeCover(lat,lon,year,tree_cover_data)
                observed_data = [elevation,precipitation,t2m,tree_cover]
                logging.info(f'elevation = {elevation}, precipitation = {precipitation}, t2m = {t2m}, tree_cover = {tree_cover}')
                x = int(np.where(temp_set['lat'].data == lat)[0])
                lat_range = temp_set['lat'].data[x-2:x+2]
                y = int(np.where(temp_set['lon'].data == lon)[0])
                lon_range = temp_set['lon'].data[y-2:y+2]
                suitable_coordinates = findSuitableCell(lat_range,lon_range,year,elevation_data,era_data,tree_cover_data,observed_data)
                print(suitable_coordinates)
                if(suitable_coordinates != (0,0)):
                    gpp = getGPP(suitable_coordinates[0], suitable_coordinates[1],year,gpp_data)
                    agb = getAGB(suitable_coordinates[0], suitable_coordinates[1],year,agb_data)
                    if(not np.isnan(gpp)):
                        total_gpp = total_gpp + ( gpp * 100000 * 60 * 60 * 24 * 365) #kgC/100km/year
                    if(not np.isnan(agb)):
                        total_agb = total_agb + ((agb * 100000) / (0.5*1.222))
                logging.info(f'total_gpp = {total_gpp}')
                logging.info(f'total agb = {total_agb}')
                logging.info('\n')
        csv_writer.writerow([year,total_gpp,total_agb])
    print('runtime: %f seconds' % (time.time() - start))
    output.close()

