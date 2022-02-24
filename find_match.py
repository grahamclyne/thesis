import xarray as xr
import time
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Point
import logging
from datetime import datetime
import csv
import geopy.distance



def checkCoordinates(lat,lon,next_lat,next_lon, boreal_coordinates):
    #if three of four coordinates are in, majority of cell counts
    count = 0
    if((lat,lon) in boreal_coordinates):
        count = count + 1
    if((lat,next_lon) in boreal_coordinates):
        count = count + 1
    if((next_lat, lon) in boreal_coordinates):
        count = count + 1
    if((next_lat,next_lon) in boreal_coordinates):
        count = count + 1
    if(count > 2):
        return True
    return False

def getClimateData(lat,lon,year,climate_data,variable):
    return climate_data[variable].sel(
        year=year,
        lat=lat, 
        lon=lon, method='nearest').mean().values  

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

def findSuitableCell(lat_range,lon_range,year,elevation_data,cesm_temp_data,cesm_precip_data,tree_cover_data,observed_data):
    for lat in lat_range:
        for lon in lon_range:
            elevation = getElevation(lat,lon,elevation_data)
            precipitation = getClimateData(lat,lon,year,cesm_precip_data,'pr')
            tas = getClimateData(lat,lon,year,cesm_temp_data,'tas')
            tree_cover = getTreeCover(lat,lon,year,tree_cover_data)
            if (compareValues([elevation,precipitation,tas,tree_cover], observed_data)):
                return (lat,lon)
    return (0,0)
   

def compareValues(simulated_values, observed_values):
    for index in range(len(simulated_values)):
        diff = abs(simulated_values[index] - observed_values[index])
        if(diff > observed_values[index]):
            return False
    return True

if __name__ == "__main__":
    start = time.time()
    logging.basicConfig(filename='/home/graham/code/thesis/logs/lookup_table_'+ str(datetime.now()) + ' .log', level=logging.DEBUG)
    output = open('output.csv', 'w')
    csv_writer = csv.writer(output)

    #load data
    data_folder = '/home/graham/code/thesis/data/'
    # zips = gpd.read_file('/home/graham/code/thesis/data/boreal_reduced.shp')
    tree_cover_data = xr.open_dataset(data_folder + 'treeFrac_Lmon_CESM2_land-hist_r1i1p1f1_gn_194901-201512.nc')
    gpp_data = xr.open_dataset(data_folder + 'gpp_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc')
    cVeg_data = xr.open_dataset(data_folder + 'cVeg_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc')
    era_precip_data = xr.open_dataset(data_folder + 'tp_groupedby_year.nc').load() 
    era_temp_data = xr.open_dataset(data_folder + 't2m_groupedby_year.nc').load()
    cesm_temp_data = xr.open_dataset(data_folder + 'tas_Amon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc')
    cesm_precip_data = xr.open_dataset(data_folder + 'pr_Amon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc')
    # elevation_data_e = xr.open_dataset(data_folder + 'e10g',engine='rasterio')
    elevation_data_b = xr.open_dataset(data_folder + 'b10g',engine='rasterio')
    # elevation_data_f = xr.open_dataset(data_folder + 'f10g', engine='rasterio')
    elevation_data_a = xr.open_dataset(data_folder + 'a10g', engine='rasterio')
    elevation_data = xr.merge([elevation_data_a,elevation_data_b])
    # elevation_data_1 = xr.merge([elevation_data_e,elevation_data_f])
    boreal_coordinates = []
    with open(data_folder + 'boreal_non_reduced_coordinates.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            boreal_coordinates.append((float(row[0]),float(row[1])))

    logging.info('data loaded.')



    #group by year
    cesm_temp_data = cesm_temp_data.groupby('time.year').mean('time')
    cesm_precip_data = cesm_precip_data.groupby('time.year').mean('time')
    gpp_data = gpp_data.groupby('time.year').mean('time')
    cVeg_data = cVeg_data.groupby('time.year').mean('time')
    tree_cover_data = tree_cover_data.groupby('time.year').mean('time')
    #prepare data
    cesm_temp_data = cesm_temp_data.isel(lat=np.logical_and(cesm_temp_data.lat<80,cesm_temp_data.lat > 50))
    cesm_temp_data = cesm_temp_data.isel(lon=np.logical_and(cesm_temp_data.lon-360>-170,cesm_temp_data.lon-360 <-50))
    cesm_temp_data = cesm_temp_data.isel(year=cesm_temp_data.year>1983)

    # tree_cover_2000 = rasterio.open(data_folder + 'Hansen_GFC-2020-v1.8_treecover2000_60N_130W.tif')
    # tree_cover_loss_year = rasterio.open(data_folder + 'Hansen_GFC-2020-v1.8_lossyear_60N_130W.tif')
    number_of_grid_cells = 0 
    number_of_grid_cells_mapped = 0
    for year in cesm_temp_data['year'].values:    
        total_gpp = 0
        total_agb = 0
        for lat_index in range(len(cesm_temp_data['lat'].values) - 1):
            for lon_index in range(len(cesm_temp_data['lon'].values) - 1):
                lat = cesm_temp_data['lat'].data[lat_index]
                lon = cesm_temp_data['lon'].data[lon_index]
                next_lat = cesm_temp_data['lat'].data[lat_index+1]
                next_lon = cesm_temp_data['lon'].data[lon_index+1]
                if year == 2015: #need to download ERA data for 2015
                    continue
                lon_scaled = lon - 360 if lon > 180 else lon
                point = Point(lon_scaled,lat)
                if(not checkCoordinates(lat,lon,next_lat,next_lon,boreal_coordinates)):
                    continue
                elevation = getElevation(lat,lon,elevation_data)
                precipitation = getERAData(lat,lon,year,era_precip_data,'tp')
                t2m = getERAData(lat,lon,year,era_temp_data,'t2m')
                tree_cover = getTreeCover(lat,lon,year,tree_cover_data)
                observed_data = [elevation,precipitation,t2m,tree_cover]
                logging.info(f'lat = {lat} lon = {lon} year={year}')
                logging.info(f'elevation = {elevation}, precipitation = {precipitation}, t2m = {t2m}, tree_cover = {tree_cover}')
                # x = int(np.where(cesm_temp_data['lat'].data == lat)[0])
                # if(len(cesm_temp_data['lat'].data) < x+1):
                lat_range = cesm_temp_data['lat'].data[lat_index-2:lat_index+1]
                # else:
                # lat_range = cesm_temp_data['lat'].data[x-2:x]
                # y = int(np.where(cesm_temp_data['lon'].data == lon)[0])
                # if(len(cesm_temp_data['lon'].data) < y+1):
                lon_range = cesm_temp_data['lon'].data[lon_index-2:lon_index+1]
                # else:
                    # lon_range = cesm_temp_data['lon'].data[y-2:y]
                squared_kilometers_long = geopy.distance.distance((lat,lon), (lat,next_lon)).km
                squared_kilometers = geopy.distance.distance((lat,lon), (next_lat,lon)).km
                area = squared_kilometers * squared_kilometers_long
                print(year,lat,lon)
                suitable_coordinates = findSuitableCell(lat_range,lon_range,year,elevation_data,cesm_temp_data,cesm_precip_data,tree_cover_data,observed_data)
                logging.info('matching cell: ' +  str(suitable_coordinates[0]) + ', ' + str(suitable_coordinates[1]))
                if(suitable_coordinates != (0,0)):
                    print('found match.')
                    gpp = getClimateData(suitable_coordinates[0], suitable_coordinates[1],year,gpp_data, 'gpp')
                    cVeg = getClimateData(suitable_coordinates[0], suitable_coordinates[1],year,cVeg_data, 'cVeg')
                    if(not np.isnan(gpp)):
                        total_gpp = total_gpp + ( gpp * 1000 * area) #kgC/100km/year
                    if(not np.isnan(cVeg)):
                        total_agb = total_agb + (cVeg * 1000 * area / (0.5*1.222)) #kg/100km
                    number_of_grid_cells_mapped = number_of_grid_cells_mapped + 1
                number_of_grid_cells = number_of_grid_cells + 1
                logging.info(f'total_gpp = {total_gpp},total agb = {total_agb}')
        csv_writer.writerow([year,total_gpp,total_agb])
    csv_writer.writerow([number_of_grid_cells, number_of_grid_cells_mapped])
    print('runtime: %f seconds' % (time.time() - start))
    output.close()

