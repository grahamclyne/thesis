from tabnanny import check
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import geopy.distance
import csv 
from find_match import checkCoordinates
def getClimateData(lat,lon,year,climate_data,variable):
    return climate_data[variable].sel(
        year=year,
        lat=lat, 
        lon=lon, method='nearest').mean().values       
 
data_folder = '/home/graham/code/thesis/data/'
cVeg_data = xr.open_dataset(data_folder + 'gpp_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc')
zips = gpd.read_file('/home/graham/code/thesis/data/boreal_reduced.shp')
file = open('output_cesm_gpp_only.csv','w')
csv_writer = csv.writer(file)
cVeg_data = cVeg_data.groupby('time.year').mean('time')
cVeg_data = cVeg_data.isel(lat=np.logical_and(cVeg_data.lat<80,cVeg_data.lat > 50))
cVeg_data = cVeg_data.isel(lon=np.logical_and(cVeg_data.lon-360>-170,cVeg_data.lon-360 <-50))
cVeg_data = cVeg_data.isel(year=cVeg_data.year>1983)
boreal_coordinates = []
with open('data/boreal_non_reduced_coordinates.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        boreal_coordinates.append((float(row[0]),float(row[1])))
for year in cVeg_data['year'].values:    
    total_gpp = 0
    total_agb = 0
    print(year)
    for lat_index in range(len(cVeg_data['lat'].values)-1):
        for lon_index in range(len(cVeg_data['lon'].values)-1):
            lat = cVeg_data['lat'].data[lat_index]
            lon = cVeg_data['lon'].data[lon_index]         
            next_lat = cVeg_data['lat'].data[lat_index+1]
            next_lon = cVeg_data['lon'].data[lon_index+1]   
            squared_kilometers_long = geopy.distance.distance((lat,lon), (lat,next_lon)).km
            squared_kilometers = geopy.distance.distance((lat,lon), (next_lat,lon)).km
            area = squared_kilometers * squared_kilometers_long
            lon_scaled = lon - 360 if lon > 180 else lon
            if(not checkCoordinates(lat,lon,next_lat,next_lon,boreal_coordinates)):
                continue
            cVeg = getClimateData(lat,lon,year,cVeg_data, 'gpp')
            if(not np.isnan(cVeg)):
                total_agb = total_agb + (cVeg * 1000 * area ) #kg/100km
    csv_writer.writerow([year,total_agb])
file.close()
          