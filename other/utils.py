import xarray as xr
from pyproj import Transformer
import rioxarray
from shapely.geometry import Polygon
from pyproj import Geod
from shapely.geometry import mapping 
import geopandas as gpd
from shapely.ops import transform
import pyproj
import numpy as np
import config
import csv 



def readCoordinates(file_name,is_grid_file) -> list:
    coordinates = []
    with open(f'{config.DATA_PATH}/{file_name}', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            if(is_grid_file):
                coordinates.append(float(row[0]))
            else:
                coordinates.append((float(row[0]),float(row[1])))
    return coordinates


def clipNFIS(nfis_tif,lat,lon,next_lat,next_lon) -> xr.DataArray:
        #this is a hack because need to get conical coordinates (at least smaller size) before clipping shapefile, otherwise takes too long

    transformer =Transformer.from_crs('epsg:4326','epsg:3978')
    x1,y1 = transformer.transform(lat,lon)
    x2,y2 = transformer.transform(next_lat,next_lon)
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
    poly = Polygon([[lon,lat],[next_lon,lat],[next_lon,next_lat],[lon,next_lat],[lon,lat]])
    wgs84 = pyproj.CRS('EPSG:4326')
    out = pyproj.CRS('EPSG:3978')
    project = pyproj.Transformer.from_crs(wgs84, out, always_xy=True).transform
    poly_proj = transform(project, poly)
    projected_poly = gpd.GeoDataFrame(index=[0], crs='epsg:3978', geometry=[poly_proj])
    sl.rio.write_crs("epsg:3978", inplace=True)
    clipped = sl.rio.clip(projected_poly.geometry.apply(mapping), projected_poly.crs,drop=True)
    return clipped


def getCoordinates(latlon:tuple,latitudes:list,longitudes:list):
    lat = latlon[0]
    lon = latlon[1]
    print(lat,lon)
    print(latitudes)
    next_lat = latitudes[latitudes.index(lat) + 1]
    next_lon = longitudes[longitudes.index(lon) + 1]
    return lat,lon,next_lat,next_lon









def countNFIS(nfis_tif:xr.Dataset,land_cover_classes:list) -> float:
    tree_coverage = nfis_tif.where(np.isin(nfis_tif.data,land_cover_classes)) #should forested wetland, 81, be included? 
    tree_coverage = tree_coverage.groupby('x')
    tree_coverage = tree_coverage.count('y')
    tree_coverage = tree_coverage.sum()
    return tree_coverage.values / nfis_tif.size * 100


def eraYearlyAverage(era,lat,lon,next_lat,next_lon,year):
    era = era.groupby('time.year').mean()
    era = era.isel(
    year=year - era.year.min().values.item(), #this only works if first year of data is 1984
    latitude=np.logical_and(era.latitude >= lat,era.latitude <= next_lat), 
    longitude=np.logical_and(era.longitude >= lon, era.longitude <= next_lon)
    )
    return (era.sum() / (len(era.latitude) * len(era.longitude)))[list(era.keys())[0]].values.item()



def getArea(lat,lon,next_lat,next_lon) -> float:
    #returns metre squared
    poly = Polygon([(lon,lat),(next_lon,lat),(lon,next_lat),(next_lat,next_lon),(lon,lat)])
    #put in counterclockwise rotation otherwise does not work
    geod = Geod(ellps="WGS84") #assume ellipsoid here
    #abs value, could be negative depending on orientation?
    area = abs(geod.geometry_area_perimeter(poly)[0])
    return area
