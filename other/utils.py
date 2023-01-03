import xarray as xr
from pyproj import Transformer,CRS,Geod
from shapely.geometry import Polygon,mapping
import geopandas as gpd
from shapely.ops import transform
import numpy as np
import other.config as config
import csv 
import cftime
import pandas as pd 
import geopandas

def readCoordinates(file_path:str,is_grid_file:bool) -> list:
    coordinates = []
    with open(f'{file_path}', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if(is_grid_file):
                coordinates.append(float(row[0]))
            else:
                coordinates.append((float(row[0]),float(row[1])))
    return coordinates

def clipNFIS(nfis_tif:xr.Dataset,lat:float,lon:float,next_lat:float,next_lon:float) -> xr.DataArray:
        #this is a hack because need to get conical coordinates (at least smaller size) before clipping shapefile, otherwise takes too long

    transformer =Transformer.from_crs('epsg:4326','epsg:3978')
    x1,y1 = transformer.transform(lat,lon)
    x2,y2 = transformer.transform(next_lat,next_lon)
    #extra area to ensure clipping does not clip data we need
    bounds = 50000
    x_min = np.amax([nfis_tif['x'].min(),x1-bounds])
    x_max = np.amin([nfis_tif['x'].max(),x2+bounds])
    y_min = np.amax([nfis_tif['y'].min(),y1-bounds])
    y_max = np.amin([nfis_tif['y'].max(),y2+bounds])
    sl = nfis_tif.isel(
        x=(nfis_tif.x >= x_min) & (nfis_tif.x < x_max),
        y=(nfis_tif.y >= y_min) & (nfis_tif.y < y_max),
        band=0
        )
    poly = Polygon([[lon,lat],[next_lon,lat],[next_lon,next_lat],[lon,next_lat]])
    wgs84 = CRS('EPSG:4326')
    out = CRS('EPSG:3978')
    project = Transformer.from_crs(wgs84, out, always_xy=True).transform
    poly_proj = transform(project, poly)
    projected_poly = gpd.GeoDataFrame(index=[0], crs='epsg:3978', geometry=[poly_proj])
    sl.rio.write_crs("epsg:3978", inplace=True)
    clipped = sl.rio.clip(projected_poly.geometry.apply(mapping), projected_poly.crs,drop=True)
    #this leaves zeroes still in, need to compensate for this later by counting the zeroes
    return clipped

def getCoordinates(latlon:tuple,latitudes:list,longitudes:list):
    lat = latlon[0]
    lon = latlon[1]
    latitudes = list(map(lambda x:round(x,7),latitudes))
    longitudes = list(map(lambda x:round(x,7),longitudes))
    lat = round(lat,7)
    lon = round(lon,7)
    next_lat = latitudes[latitudes.index(lat) + 1]
    next_lon = longitudes[longitudes.index(lon) + 1]
    return lat,lon,next_lat,next_lon

def getArea(lat,lon,next_lat,next_lon) -> float:
    #returns metre squared
    poly = Polygon([(lon,next_lat),(lon,lat),(next_lon,lat),(next_lon,next_lat)])
    #put in counterclockwise rotation otherwise does not work
    geod = Geod(ellps="WGS84") #assume ellipsoid here
    #abs value, could be negative depending on orientation?
    area = geod.geometry_area_perimeter(poly)
    # print(area)
    return area[0]

#convert 0-360 to 180 lons, and re-sort 
def scaleLongitudes(dataset:xr.Dataset,lon_variable:str) -> xr.Dataset:
    dataset[lon_variable] = np.where(dataset[lon_variable]> 180,dataset[lon_variable] - 360, dataset[lon_variable]) 
    return dataset.sortby(dataset.lon)


def clipWithShapeFile(netcdf_file,variable,shape_file):
    netcdf_file = netcdf_file[variable]
    netcdf_file.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    netcdf_file.rio.write_crs("epsg:4326", inplace=True)
    clipped = netcdf_file.rio.clip([shape_file.geometry.apply(mapping)[0]], shape_file.crs,drop=True)
    return clipped


#split temperatures into seasonal temps
def seasonalAverages(netcdf_file:xr.Dataset, variable:str,shape_file:geopandas.GeoDataFrame,upstream:str) -> list:
    netcdf_file = scaleLongitudes(netcdf_file, 'lon')
    years = np.unique(netcdf_file['time.year'].data)
    netcdf_file = clipWithShapeFile(netcdf_file,variable,shape_file)
    output = [[] for _ in range(4)]
    #stack each seasonal avg of each year
    for year in years:
        if(upstream == 'CESM'):
            start_time = cftime.DatetimeNoLeap(year, 1, 1, 1, 0, 0, 0,has_year_zero=True)
            end_time =cftime.DatetimeNoLeap(year+1, 1, 1, 1, 0, 0, 0,has_year_zero=True)
        elif(upstream == 'ERA'):
            start_time = np.datetime64(f'{year}-01-01')
            end_time =np.datetime64(f'{year+1}-01-01')
        year_slice = netcdf_file.loc[dict(time=slice(start_time,end_time))]
        seasonal_data = year_slice.groupby('time.season').mean()
        seasons = seasonal_data.season.data
        for season_index in range(len(seasonal_data.season.data)):
            single_season_data = seasonal_data.sel(season=seasons[season_index]).data.reshape(1,-1)
            output[season_index].append(single_season_data)
    #concatenate values together
    output = [np.concatenate(x,axis=1).reshape(-1,1) for x in output]
    #remove null values
    # output = [np.expand_dims(x[~np.isnan(x)].compute_chunk_sizes(),axis=1) for x in output]
    return output


def getRollingWindow(dataframe:pd.DataFrame,seq_len:int,inputs) -> pd.DataFrame:
    dataframe = dataframe[inputs]
    windows = pd.DataFrame()
    for window in dataframe.rolling(window=seq_len):
        if(len(window) == seq_len):
            windows = pd.concat([windows,window])
    return windows
