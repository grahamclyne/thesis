import multiprocessing
import csv 
import rioxarray
import config
import time
import multiprocessing
from pyproj import Transformer
import numpy as np
from shapely.geometry import Polygon
import rioxarray
from shapely.ops import transform
import xarray as xr
import pyproj
from shapely.geometry import mapping 
import geopandas as gpd

def getCoordinates(latlon:tuple,latitudes:list,longitudes:list):
    lat = latlon[0]
    lon = latlon[1]
    next_lat = latitudes[latitudes.index(lat) + 1]
    next_lon = longitudes[longitudes.index(lon) + 1]
    return lat,lon,next_lat,next_lon



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

    print(poly)
    print(lat,lon,next_lat,next_lon)
    wgs84 = pyproj.CRS('EPSG:4326')
    out = pyproj.CRS('EPSG:3978')

    project = pyproj.Transformer.from_crs(wgs84, out, always_xy=True).transform
    poly_proj = transform(project, poly)
    print(poly_proj)
    projected_poly = gpd.GeoDataFrame(index=[0], crs='epsg:3978', geometry=[poly_proj])
    sl.rio.write_crs("epsg:3978", inplace=True)
    print(sl)
    clipped = sl.rio.clip(projected_poly.geometry.apply(mapping), projected_poly.crs,drop=True)
    print(clipped)
    return clipped

def getRow(nfis_tif,lat,lon,next_lat,next_lon):
    print(lat,lon,multiprocessing.current_process())
    agb = clipNFIS(nfis_tif,lat,lon,next_lat,next_lon).sum()
    return [agb,lat,lon]

if __name__ == "__main__":
    nfis_tif = rioxarray.open_rasterio(f'{config.NFIS_PATH}/CA_forest_total_biomass_2015_NN/CA_forest_total_biomass_2015.tif',lock=False)
    start_time = time.time()
    boreal_coordinates = []
    ordered_latitudes = []
    ordered_longitudes = []

    with open(f'{config.DATA_PATH}/boreal_latitudes_longitudes.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            boreal_coordinates.append((float(row[0]),float(row[1])))

    with open(f'{config.DATA_PATH}/grid_latitudes.csv',newline='') as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        for row in reader:
            ordered_latitudes.append(float(row[0]))

    with open(f'{config.DATA_PATH}/grid_longitudes.csv',newline='') as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        for row in reader:
            ordered_longitudes.append(float(row[0]))


    observable_rows = open(f'{config.DATA_PATH}/nfis_agb.csv','w')
    writer = csv.writer(observable_rows)
    writer.writerow(['agb','lat','lon'])

    # print(boreal_coor)
    x = iter(boreal_coordinates)
    p = multiprocessing.Pool(5)
    with p:
        for i in range(int(len(boreal_coordinates))):
            lat,lon,next_lat,next_lon = getCoordinates(next(x),ordered_latitudes,ordered_longitudes)
            print(lat,lon)
            row = getRow(nfis_tif,lat,lon,next_lat,next_lon)
            print(row)
            #beware, failed child processes do not give error by default
            p.apply_async(getRow,[nfis_tif,lat,lon,next_lat,next_lon],callback = writer.writerow)
            # x.get() use this line for debugging
        p.close()
        p.join()
    nfis_tif.close()
    observable_rows.close()
    duration = time.time() - start_time
    print(f'Completed in {duration} seconds.')
