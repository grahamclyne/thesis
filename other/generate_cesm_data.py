import xarray as xr
import other.config as config
import time
from shapely.geometry import mapping
import geopandas
import numpy as np
import os
from other.utils import scaleLongitudes


def netcdfToNumpy(netcdf_file,variable,shape_file,getUniqueKey):
    netcdf_file = scaleLongitudes(netcdf_file, 'lon')
    netcdf_file = netcdf_file.groupby('time.year').mean()
    netcdf_file = netcdf_file[variable]
    netcdf_file.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    netcdf_file.rio.write_crs("epsg:4326", inplace=True)
    clipped = netcdf_file.rio.clip([shape_file.geometry.apply(mapping)[0]], shape_file.crs,drop=True)
    print(clipped)
    df = clipped.to_dataframe().reset_index()
    df = df.dropna(0)
    print(df)
    if(getUniqueKey):
        return df[['year','lat','lon',variable]].values
    else:
        return df[[variable]].values



def combineNetCDFs(file_paths:list,shape_file:geopandas.GeoDataFrame) -> np.ndarray :
    out = np.empty(0)
    completed = []
    getUniqueKey = True
    for file_index in range(len(file_paths)):
        ds = xr.open_dataset(f'{config.CESM_PATH}/{file_paths[file_index]}',engine='netcdf4')
        var = file_paths[file_index].split('_')[0]  
        if(var in completed):
            continue      
        if(var == 'tsl'):
            #indices 0-3 cover the top 10cm
            ds = ds.isel(depth=slice(0,3)).groupby('time').sum('depth')
        #search rest of files for same variable, need to check if file is split into multiple time ranges
        for file_index1 in range(file_index,len(file_paths)):
            if var + '_' in file_paths[file_index1] and file_paths[file_index1] != file_paths[file_index]:
                other_ds = xr.open_dataset(f'{config.CESM_PATH}/{file_paths[file_index1]}',engine='netcdf4')
                if(var == 'tsl'):
                    other_ds = other_ds.isel(depth=slice(0,3)).groupby('time').sum('depth')
                ds = xr.merge([ds,other_ds])
                other_ds.close()
        ds = ds.fillna(0)
        arr = netcdfToNumpy(ds,var,shape_file,getUniqueKey)
        getUniqueKey = False
        completed.append(var)
        #if array is empty, set first variable
        if(len(out) == 0):
            out = arr.reshape(-1,4)
        else:
            out = np.concatenate((out,arr[:,0].reshape(-1,1)),1)
        ds.close()
        print(out)
    #append year,lat,lon for index - can append them from first file as they should all be the same
    # for file in os.system(f'ls {config.CESM_PATH}/pr_*'):
    #     ds_temp = xr.open_dataset(file,engine='netcdf4')
    #     ds = xr.merge([ds,ds_temp])
    # years = netcdfToNumpy(ds,'year',shape_file)
    # lat = netcdfToNumpy(ds,'lat',shape_file)
    # lon = netcdfToNumpy(ds,'lon',shape_file)
    # out = np.concatenate((out,years),1)
    # out = np.concatenate((out,lat),1)
    # out = np.concatenate((out,lon),1)
    return out,completed 


# num of rows outputted should be # of years (31) * length of boreal coordinates (788) = 24428
if __name__ == '__main__':
    start_time = time.time()

    os.chdir(f'{config.CESM_PATH}')
    input_files = os.listdir()

    shape_file = geopandas.read_file(f'{config.SHAPEFILE_PATH}/NIR2016_MF.shp', crs="epsg:4326")

    combined,header = combineNetCDFs(input_files,shape_file)
    # header = list(map(lambda x: x.split('_')[0],input_files))
    header = ['year','lat','lon'] + header
    header = ','.join(header)
    print(header)
    np.savetxt(f'{config.DATA_PATH}/cesm_data.csv',np.asarray(combined),delimiter=',',header=header)
    duration = time.time() - start_time
    print(f'Completed in {duration} seconds.')
