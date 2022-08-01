from sklearn import tree
import xarray as xr
import numpy as np
import cfgrib
import csv
import cftime
import matplotlib.pyplot as plt
import datetime
boreal_coordinates = []
latitudes = []
longitudes = []


def filter_coordinates_era(boreal_coordinates,dataset,variable):
    dataset = dataset.isel(latitude=np.logical_and(dataset.latitude<80,dataset.latitude> 40))
    dataset = dataset.isel(longitude=np.logical_and(dataset.longitude>-170,dataset.longitude <-50))
    dataset = dataset.isel(time=dataset.time >= np.datetime64('1984-01-01'))  
    new = xr.Dataset()
    for (lat,lon) in boreal_coordinates:
        temp_ds = dataset[variable].isel(
        latitude=np.logical_and(dataset.latitude >= lat,dataset.latitude <= lat+0.9), 
        longitude=np.logical_and(dataset.longitude >= lon, dataset.longitude <= lon+1.25)
        ) 
        if(len(temp_ds['latitude'].data) > 0 and len(temp_ds['longitude'].data >0)):
            new = xr.merge([new,temp_ds])
    return new.groupby('time').mean(...)



def filter_coordinates(boreal_coordinates,dataset,variable):
    dataset = dataset.isel(lat=np.logical_and(dataset.lat<80,dataset.lat> 40))
    dataset = dataset.isel(lon=np.logical_and(dataset.lon-180>-170,dataset.lon-180 <-50))
    dataset = dataset.isel(time=dataset.time >= cftime.DatetimeNoLeap(1984, 1, 1))  
    for latitude in dataset['lat'].data:
        for longitude in dataset['lon'].data:
            if((latitude,longitude - 180) in boreal_coordinates):
                continue
            else:
                dataset[variable].loc[dict(lat=latitude,lon=longitude)] = np.NaN
    return dataset.groupby('time').mean(...)



if __name__ == '__main__':
    boreal_coordinates = []
    with open('/home/graham/code/thesis/canadian_boreal_coordinates.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(spamreader) #skip header
        for row in spamreader:
            boreal_coordinates.append((float(row[0]),float(row[1])))

    tp_era_1984_2001 = xr.open_dataset('/home/graham/Downloads/era_temp_precip_1984_2015.grib',filter_by_keys={'stepType': 'avgas'})
    tp_era_2002_2015 = xr.open_dataset('/home/graham/Downloads/era_temp_precip_1984_2015.grib',filter_by_keys={'stepType':'avgad'})
    t2m_era_1984_2015 = xr.open_dataset('/home/graham/Downloads/era_temp_precip_1984_2015.grib',filter_by_keys={'stepType': 'avgid'})
    tp = xr.concat([tp_era_1984_2001,tp_era_2002_2015],dim='time')

    data_folder = 'data/'
    tree_cover_data = xr.open_dataset(data_folder + 'treeFrac_Lmon_CESM2_land-hist_r1i1p1f1_gn_194901-201512.nc')
    cVeg_data = xr.open_dataset(data_folder + 'cVeg_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc')
    cesm_temp_data = xr.open_dataset(data_folder + 'tas_Amon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc')
    cesm_precip_data = xr.open_dataset(data_folder + 'pr_Amon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc')


    tp = xr.concat([tp_era_1984_2001,tp_era_2002_2015],dim='time')

    filter_coordinates(boreal_coordinates,tree_cover_data,'treeFrac').to_netcdf('cesm_tree_cover_canadian_boreal.nc')
    filter_coordinates(boreal_coordinates,cVeg_data,'cVeg').to_netcdf('cesm_cveg_canadian_boreal.nc')
    filter_coordinates(boreal_coordinates,cesm_temp_data,'tas').to_netcdf('cesm_temp_canadian_boreal.nc')
    filter_coordinates(boreal_coordinates,cesm_precip_data,'pr').to_netcdf('cesm_precip_canadian_boreal.nc')
    filter_coordinates_era(boreal_coordinates,t2m_era_1984_2015,'t2m').to_netcdf('era_t2m_canadian_boreal.nc')
    filter_coordinates_era(boreal_coordinates,tp,'tp').to_netcdf('era_tp_canadian_boreal.nc')


