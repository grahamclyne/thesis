import xarray as xr
import time
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import logging
from datetime import datetime
import csv
import geopy.distance
import rioxarray
from pyproj import Transformer

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
        latitude=np.logical_and(era_data.latitude >= lat,era_data.latitude <= lat+0.9), 
        longitude=np.logical_and(era_data.longitude >= lon, era_data.longitude <= lon+1.25)
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

def getObservedTreeCover(lat,next_lat,lon,next_lon,obs_tree,transformer):
    lat,lon = transformer.transform(lat,lon)
    next_lat,next_lon = transformer.transform(next_lat,next_lon)
    sl = obs_tree.isel(
        x=np.logical_and(obs_tree.x >= lat, obs_tree.x <= next_lat),
        y=np.logical_and(obs_tree.y >= lon, obs_tree.y <= next_lon),
        band=0
        )
    if (sl.size == 0):
        return 0
    tree_coverage = sl.where(np.isin(sl.data,[230,220,210,81]))
    tree_coverage = tree_coverage.groupby('x')
    tree_coverage = tree_coverage.count('y')
    tree_coverage = tree_coverage.sum()
    return tree_coverage.values / sl.size * 100

def findSuitableCell(lat_range,lon_range,year,elevation_data,cesm_temp_data,cesm_precip_data,tree_cover_data,observed_data):
    for lat in lat_range:
        for lon in lon_range:
            elevation = getElevation(lat,lon,elevation_data)
            precipitation = getClimateData(lat,lon,year,cesm_precip_data,'pr')
            tas = getClimateData(lat,lon,year,cesm_temp_data,'tas')
            tree_cover = getTreeCover(lat,lon,year,tree_cover_data)
            logging.info(f'CLIMATE elevation = {elevation}, precipitation = {precipitation}, tas = {tas}, tree_cover = {tree_cover}')

            if (compareValues([elevation,precipitation,tas,tree_cover], observed_data)):
                return (lat,lon)
    return (0,0)
   

def compareValues(simulated_values, observed_values):
    for index in range(len(simulated_values)):
        diff = abs(simulated_values[index] - observed_values[index])
        
        if(index != 3 and diff > observed_values[index]): #terrible workaround for chekcing if treecover within 5%
            return False
        elif(diff > 5):
            return False
    return True

if __name__ == "__main__":
    start = time.time()
    logging.basicConfig(filename='logs/lookup_table_'+ str(datetime.now()) + '.log', level=logging.INFO)
    output = open('output.csv', 'w')
    csv_writer = csv.writer(output)
    #need to check operating system 

    #load data
    data_folder = '/Users/gclyne/'
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
    transformer =Transformer.from_crs(4326,3978)

    number_of_grid_cells = 0 
    number_of_grid_cells_mapped = 0
    for year in cesm_temp_data['year'].values:    
        total_gpp = 0
        total_agb = 0
        observed_tree_cover_data = rioxarray.open_rasterio('/Users/gclyne/Downloads/CA_forest_VLCE2_' + str(year) + '/CA_forest_VLCE2_' + str(year) + '.tif')
        for lat_index in range(len(cesm_temp_data['lat'].values) - 1):
            for lon_index in range(len(cesm_temp_data['lon'].values) - 1):
                lat = cesm_temp_data['lat'].data[lat_index]
                lon = cesm_temp_data['lon'].data[lon_index]
                next_lat = cesm_temp_data['lat'].data[lat_index+1]
                next_lon = cesm_temp_data['lon'].data[lon_index+1]
                lon_scaled = lon - 360 if lon > 180 else lon # this is not the right formula, does not cover edge cases eg. 178 goes to -180
                point = Point(lon_scaled,lat)
                if(not checkCoordinates(lat,lon,next_lat,next_lon,boreal_coordinates)):
                    continue
                print(year,lat,lon)

                elevation = getElevation(lat,lon_scaled,elevation_data)
                precipitation = getERAData(lat,lon_scaled,year,era_precip_data,'tp') / 100
                t2m = getERAData(lat,lon_scaled,year,era_temp_data,'t2m')
                tree_cover = getObservedTreeCover(lat,next_lat,lon_scaled,lon_scaled + 1.25,observed_tree_cover_data,transformer)
                observed_data = [elevation,precipitation,t2m,tree_cover]
                logging.info(f'lat = {lat} lon = {lon} year={year}')
                logging.info(f'elevation = {elevation}, precipitation = {precipitation}, t2m = {t2m}, tree_cover = {tree_cover}')
                lat_range = cesm_temp_data['lat'].data[lat_index-2:lat_index+1]
                lon_range = cesm_temp_data['lon'].data[lon_index-2:lon_index+1]
                squared_kilometers_long = geopy.distance.distance((lat,lon), (lat,next_lon)).km
                squared_kilometers = geopy.distance.distance((lat,lon), (next_lat,lon)).km
                area = squared_kilometers * squared_kilometers_long
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
        observed_tree_cover_data.close()
    csv_writer.writerow([number_of_grid_cells, number_of_grid_cells_mapped])
    print('runtime: %f seconds' % (time.time() - start))
    output.close()

