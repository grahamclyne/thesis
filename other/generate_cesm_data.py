import xarray as xr
import config
import time
from shapely.geometry import mapping
import geopandas
import numpy as np
import os
def netcdfToNumpy(netcdf_file,variable,shape_file):
    netcdf_file['lon'] = netcdf_file['lon'] - 360 if np.any(netcdf_file['lon'] > 180) else netcdf_file['lon']
    netcdf_file = netcdf_file.groupby('time.year').mean()
    # netcdf_file = netcdf_file.sel(year=netcdf_file.year>=1949)
    netcdf_file = netcdf_file[variable]
    netcdf_file.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    netcdf_file.rio.write_crs("epsg:4326", inplace=True)
    clipped = netcdf_file.rio.clip([shape_file.geometry.apply(mapping)[0]], shape_file.crs,drop=True)

    df = clipped.to_dataframe().reset_index()
    df = df[df[variable].notna()]
    return df[[variable,'year','lat','lon']].values


def combineNetCDFs(file_paths:list,shape_file:geopandas.GeoDataFrame) -> np.ndarray :
    # canada_shape_file = geopandas.read_file(canada_shape_file_path)

    out = np.array([])
    years = []
    for file in file_paths:
        ds = xr.open_dataset(f'{config.CESM_PATH}/{file}',engine='netcdf4')
        #need to check if file is split into two time ranges
        var = file.split('_')[0]
        years = file.split('_')[0]
        for file1 in file_paths:
            if var in file1 and file1 != file:
                other_ds = xr.open_dataset(f'{config.CESM_PATH}/{file1}',engine='netcdf4')
                ds = xr.merge([ds,other_ds])
        arr = netcdfToNumpy(ds,var,shape_file)
        #get year for only one column, they will all be the same
        years = arr[:,1].reshape(-1,1)
        lat = arr[:,2].reshape(-1,1)
        lon = arr[:,3].reshape(-1,1)
        print(arr)
        print(out)
        if(len(out) == 0):
            out = arr[:,0].reshape(-1,1)
        else:
            out = np.concatenate((out,arr[:,0].reshape(-1,1)),1)
    out = np.concatenate((out,years),1)
    out = np.concatenate((out,lat),1)
    out = np.concatenate((out,lon),1)

    return out 

# num of rows outputted should be # of years (31) * length of boreal coordinates (788) = 24428
if __name__ == '__main__':
    start_time = time.time()

    os.chdir(f'{config.CESM_PATH}')
    input_files = os.listdir()

    shape_file = geopandas.read_file(f'{config.SHAPEFILE_PATH}/NIR2016_MF.shp', crs="epsg:4326")

    combined = combineNetCDFs(input_files,shape_file)
    header = ','.join(list(map(lambda x: x.split('_')[0],input_files)) + ['years','latitude','longitude'])
    np.savetxt(f'{config.DATA_PATH}/cesm_data.csv',np.asarray(combined),delimiter=',',header=header)
    duration = time.time() - start_time
    print(f'Completed in {duration} seconds.')
