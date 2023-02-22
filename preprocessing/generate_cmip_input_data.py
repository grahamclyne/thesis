import xarray as xr
import time
import geopandas
import numpy as np
import pandas as pd
from preprocessing.utils import scaleLongitudes, seasonalAverages,clipWithShapeFile,getArea

import hydra
from omegaconf import DictConfig

def netcdfToNumpy(netcdf_file,variable,shape_file,getUniqueKey):
    netcdf_file = scaleLongitudes(netcdf_file)
    netcdf_file = netcdf_file.groupby('time.year').mean()
    clipped = clipWithShapeFile(netcdf_file,variable,shape_file)
    df = clipped.to_dataframe().reset_index()
    if(getUniqueKey):
        return df[['year','lat','lon',variable]].values.reshape(-1,4)
    else:
        return df[[variable]].values.reshape(-1,1)


def CESMVariables(variant,cfg):
    shape_file = geopandas.read_file(f'{cfg.environment.path.shapefiles}/NIR2016_MF.shp', crs="epsg:4326")
    out = []
    header = cfg.model.raw_cmip_variables
    getYearLatLon = True
    #replace tas with seasons
    header = header[:header.index('tas')] + ['tas_DJF','tas_JJA','tas_MAM','tas_SON'] + header[header.index('tas')+1:]     
    for var in cfg.model.raw_cmip_variables:
        ds = xr.open_mfdataset(f'{cfg.environment.path.cesm}/{var}*r{variant}i1p1f1*.nc',parallel=True)
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
    df['lat'] = df['lat'].round(7)
    df['lon'] = df['lon'].round(7)
    return df,df.columns     


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    start_time = time.time()
    full_df = pd.DataFrame()
    for variant in range(1,8):
        combined,header = CESMVariables(variant,cfg)
        df,header = transformData(combined,header)
        header = ','.join(header)
        #drop rows that are all null for input variables
        df = df.dropna(how='any')
        #convert ndarray to array and save 
        df['variant'] = variant
        full_df = pd.concat([full_df,df],ignore_index=True)
    full_df['area'] =  full_df.apply(lambda x: getArea(x['lat'],x['lon']),axis=1)

    header = header + ',area'
    np.savetxt(f'{cfg.data}/cesm_data_variant.csv',np.asarray(full_df),delimiter=',',header=header)
    duration = time.time() - start_time
    print(f'Completed in {duration} seconds.')

main()