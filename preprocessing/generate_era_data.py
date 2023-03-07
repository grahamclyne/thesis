import os
import time
import geopandas 
import numpy as np
from preprocessing.utils import netcdfToNumpy
import xarray as xr
import pandas as pd
from preprocessing.constants import RAW_ERA_VARIABLES
from preprocessing.utils import seasonalAverages
import hydra
from omegaconf import DictConfig





# def combineERANetcdf(file_paths:list,shape_file:geopandas.GeoDataFrame,cfg) -> np.ndarray :
#     out = np.empty(0)
#     getUniqueKey = True
#     headers = []
#     for file_index in range(len(file_paths)):
#         print(file_paths[file_index])
#         ds = xr.open_dataset(f'{cfg.environment.era_data}/reprojected/{file_paths[file_index]}',engine='netcdf4')
#         var = list(ds.keys())[0]
#         headers.append(var)
#         ds = ds.fillna(0)
#         arr = netcdfToNumpy(ds,var,shape_file,getUniqueKey)
#         getUniqueKey = False
#         #if array is empty, set first variable
#         if(len(out) == 0):
#             out = arr.reshape(-1,4)
#         else:
#             print(arr[:,0].reshape(-1,1))
#             out = np.concatenate((out,arr[:,0].reshape(-1,1)),1)
#         ds.close()
#     return out,headers


def ERAVariables(shape_file,cfg):
    getYearLatLon = True
    out = []
    headers = []
    for raw_variable in RAW_ERA_VARIABLES:
        ds = xr.open_dataset(f'{cfg.environment.era_data}/reprojected/reprojected_{raw_variable}.nc',engine='netcdf4')
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


def reprojectNetCDF(file_name,cfg):
    os.system(f'cdo remapbil,{cfg.environment.era_data}/mygrid {cfg.environment.era_data}/{file_name} {cfg.environment.era_data}/reprojected/reprojected_{file_name}')


def transformData(array,header):
    df = pd.DataFrame(array)
    df.columns = header
    df['lai'] = df['lai_lv'] + df['lai_hv']
    print(df['tp'].describe())
    df['tp'] = (df['tp']*1000) / (60*60*24)#convert from  M to kg/m2/s, using stream=moda, see https://confluence.ecmwf.int/display/CKB/ERA5-Land%3A+data+documentation#ERA5Land:datadocumentation-monthlymeansMonthlymeans 
    print(df['tp'].describe())
    df['ro'] = (df['ro'] * 1000) / (60*60*24)
    df['evabs'] = (df['evabs'] * 1000)/ (24*60*60) * -1
    df['swvl1'] = df['swvl1'] * 100
    return df 


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    start_time = time.time()
    for file in RAW_ERA_VARIABLES:
        if(not os.path.exists(f'{cfg.environment.era_data}/reprojected/reprojected_{file}.nc')):
            reprojectNetCDF(file,cfg)
    shape_file = geopandas.read_file(f'{cfg.environment.shapefiles}/NIR2016_MF.shp', crs="epsg:4326")
    combined,header = ERAVariables(shape_file,cfg)
    df = transformData(combined,header)
    print(df['tp'].describe())
    print(df.columns)
    np.savetxt(f'{cfg.environment.era_data}/era_data.csv',np.asarray(df),delimiter=',',header=','.join(df.columns))
    print(f'Completed in {time.time() - start_time} seconds.')

main()