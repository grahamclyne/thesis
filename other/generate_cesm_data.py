import xarray as xr
import config
import time
from shapely.geometry import mapping
import geopandas
import numpy as np
import rioxarray 

def netcdfToNumpy(netcdf_file,variable,shape_file,canada_shape_file):
    netcdf_file['lon'] = netcdf_file['lon'] - 360 if np.any(netcdf_file['lon'] > 180) else netcdf_file['lon']
    netcdf_file = netcdf_file.groupby('time.year').mean()
    netcdf_file = netcdf_file.sel(year=netcdf_file.year>=1949)
    netcdf_file = netcdf_file[variable]
    netcdf_file.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    netcdf_file.rio.write_crs("epsg:4326", inplace=True)
    clipped = netcdf_file.rio.clip(shape_file.geometry.apply(mapping), shape_file.crs,drop=True)
    clipped = clipped.rio.clip(canada_shape_file.geometry.apply(mapping), canada_shape_file.crs,drop=True)

    df = clipped.to_dataframe().reset_index()
    df = df[df[variable].notna()]
    return df[[variable]].values


def combineNetCDFs(file_paths:list,shp_file_path:str,canada_shape_file_path) -> np.ndarray :
    shp_file = geopandas.read_file(shp_file_path, crs="epsg:4326")
    canada_shape_file = geopandas.read_file(canada_shape_file_path)

    out = np.array([])
    for file in file_paths:
        ds = xr.open_dataset(config.CESM_PATH + '/' + file,engine='netcdf4')
        var = file.split('_')[0]
        arr = netcdfToNumpy(ds,var,shp_file,canada_shape_file)
        if(len(out) == 0):
            out = arr
        else:
            out = np.concatenate((out,arr),1)
    return out 

# num of rows outputted should be # of years (31) * length of boreal coordinates (788) = 24428
if __name__ == '__main__':
    start_time = time.time()

    input_files =['lai_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc',
        'pr_Amon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc',
        'tas_Amon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc',
        'treeFrac_Lmon_CESM2_land-hist_r1i1p1f1_gn_194901-201512.nc',
        'cVeg_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc',
        'cSoil_Emon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc',
        'cLitter_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc',
        'cCwd_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc']
    canada_shape_file_path = f'{config.SHAPEFILE_PATH}/lpr_000b21a_e/lpr_000b21a_e.shp'
    boreal_shape_file_path = f'{config.SHAPEFILE_PATH}/NABoreal.shp'
    combined = combineNetCDFs(input_files,boreal_shape_file_path,canada_shape_file_path)
    np.savetxt(f'{config.DATA_PATH}/cesm_data.csv',np.asarray(combined),delimiter=',')
    duration = time.time() - start_time
    print(f'Completed in {duration} seconds.')
