import os
import other.config as config
import time
import geopandas 
import numpy as np
from other.generate_cmip_input_data import netcdfToNumpy
import xarray as xr
import pandas as pd
from other.constants import RAW_ERA_VARIABLES
from other.utils import seasonalAverages





def combineERANetcdf(file_paths:list,shape_file:geopandas.GeoDataFrame) -> np.ndarray :
    out = np.empty(0)
    getUniqueKey = True
    headers = []
    for file_index in range(len(file_paths)):
        print(file_paths[file_index])
        ds = xr.open_dataset(f'{config.ERA_PATH}/reprojected/{file_paths[file_index]}',engine='netcdf4')
        var = list(ds.keys())[0]
        headers.append(var)
        ds = ds.fillna(0)
        arr = netcdfToNumpy(ds,var,shape_file,getUniqueKey)
        getUniqueKey = False
        #if array is empty, set first variable
        if(len(out) == 0):
            out = arr.reshape(-1,4)
        else:
            print(arr[:,0].reshape(-1,1))
            out = np.concatenate((out,arr[:,0].reshape(-1,1)),1)
        ds.close()
    return out,headers


def ERAVariables(shape_file):
    getYearLatLon = True
    out = []
    headers = []
    for raw_variable in RAW_ERA_VARIABLES:
        ds = xr.open_dataset(f'{config.ERA_PATH}/reprojected/reprojected_{raw_variable}.nc',engine='netcdf4')
        ds = ds.fillna(0)
        var = list(ds.keys())[0]
        if(var == 't2m'):
            out.extend(seasonalAverages(ds,var,shape_file,'ERA'))
            headers.extend(['tas_DJF','tas_JJA','tas_MAM','tas_SON'])
        else:
            out.append(netcdfToNumpy(ds,var,shape_file,getYearLatLon))
            if(getYearLatLon):
                headers.extend(['year','lat','lon'])
            getYearLatLon = False
            headers.append(var)
        ds.close()
    out = np.concatenate(out,axis=1)
    return out,headers


def reprojectNetCDF(file_name):
    os.system(f'cdo remapbil,{config.DATA_PATH}/mygrid {config.ERA_PATH}/{file_name} {config.ERA_PATH}/reprojected/reprojected_{file_name}')


def transformData(array,header):
    df = pd.DataFrame(array)
    df.columns = header
    df['lai'] = df['lai_lv'] + df['lai_hv']
    df['tp'] = (df['tp']*1000) / (24*60*60*30.437)
    df['ro'] = (df['ro'] * 1000) / (24*60*60*30.437)
    df['evabs'] = (df['evabs'] * 1000)/ (24*60*60*30.437) * -1
    df['swvl1'] = df['swvl1'] * 100
    return df 


if __name__ == '__main__':
    start_time = time.time()
    for file in RAW_ERA_VARIABLES:
        if(not os.path.exists(f'{config.ERA_PATH}/reprojected/reprojected_{file}.nc')):
            reprojectNetCDF(file)
    shape_file = geopandas.read_file(f'{config.SHAPEFILE_PATH}/NIR2016_MF.shp', crs="epsg:4326")
    combined,header = ERAVariables(shape_file)
    df = transformData(combined,header)
    np.savetxt(f'{config.DATA_PATH}/era_data.csv',np.asarray(df),delimiter=',',header=','.join(df.columns))
    print(f'Completed in {time.time() - start_time} seconds.')