import xarray as xr
import other.config as config
import time
import geopandas
import numpy as np
import pandas as pd
from other.utils import scaleLongitudes, seasonalAverages,clipWithShapeFile
from other.constants import MODEL_INPUT_VARIABLES, RAW_CMIP_VARIABLES


def netcdfToNumpy(netcdf_file,variable,shape_file,getUniqueKey):
    netcdf_file = scaleLongitudes(netcdf_file, 'lon')
    netcdf_file = netcdf_file.groupby('time.year').mean()
    clipped = clipWithShapeFile(netcdf_file,variable,shape_file)
    df = clipped.to_dataframe().reset_index()
    if(getUniqueKey):
        return df[['year','lat','lon',variable]].values.reshape(-1,4)
    else:
        return df[[variable]].values.reshape(-1,1)


def CESMVariables():
    shape_file = geopandas.read_file(f'{config.SHAPEFILE_PATH}/NIR2016_MF.shp', crs="epsg:4326")
    out = []
    header = RAW_CMIP_VARIABLES
    getYearLatLon = True
    #replace tas with seasons
    header = header[:header.index('tas')] + ['tas_DJF','tas_MAM','tas_JJA','tas_SON'] + header[header.index('tas')+1:]     
    for var in RAW_CMIP_VARIABLES:
        ds = xr.open_mfdataset(f'{config.CESM_PATH}/{var}*.nc',parallel=True)
        if(var == 'tas'):
            out.extend(seasonalAverages(ds,var,shape_file,'CESM'))
        else:
            if(var == 'tsl'):
                ds = ds.isel(depth=slice(0,3)).groupby('time').sum('depth') / 3
                # print(type(ds))
                # ds = ds.mean('depth')
            out.append(netcdfToNumpy(ds,var,shape_file,getYearLatLon))
            getYearLatLon = False
    header = ['year','lat','lon'] + header
    return np.concatenate(out,axis=1),header

def transformData(array,header):
    df = pd.DataFrame(array)
    df.columns = header
    df['grassCropFrac'] = df['grassFrac'] + df['cropFrac']
    return df,df.columns     

if __name__ == '__main__':
    start_time = time.time()
    combined,header = CESMVariables()
    df,header = transformData(combined,header)
    header = ','.join(header)

    #drop rows that are all null for input variables
    df = df.dropna(how='any')
    #convert ndarray to array and save 
    np.savetxt(f'{config.DATA_PATH}/cesm_data.csv',np.asarray(df),delimiter=',',header=header)
    duration = time.time() - start_time
    print(f'Completed in {duration} seconds.')
