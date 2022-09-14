import numpy as np
import xarray as xr
from pyproj import Transformer
import ee
import folium
import rioxarray

def clipNFIS(nfis_tif,lat,lon,next_lat,next_lon) -> xr.DataArray:
        #this is a hack because need to get conical coordinates (at least smaller size) before reproejcting to 4326, otherwise takes too long

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
    sl = sl.rio.reproject('EPSG:4236')

    sl = sl.isel(
        x=np.logical_and(sl.x >= lon, sl.x <= next_lon),
        y=np.logical_and(sl.y >= lat, sl.y <= next_lat)
        )
    return sl


def getCoordinates(latlon):
    lat = latlon[0]
    lon = latlon[1]
    next_lat = lat + 1
    next_lon = lon + 1
    return lat,lon,next_lat,next_lon


def getMODISLAI(lat,lon,next_lat,next_lon,year):
    try:     
        ee.Initialize()
    except:
        ee.Authenticate()
        ee.Initialize()
    b_box = ee.Geometry.BBox(lon,lat,next_lon,next_lat)
    b_box_bounds = b_box.bounds()
    start_date = ee.Date(str(year))
    end_date = start_date.advance(1,'year')
    modis = ee.ImageCollection('MODIS/061/MOD15A2H').filterDate(start_date, end_date).filterBounds(b_box_bounds).select('Lai_500m').mean()
    modis = modis.clip(b_box_bounds)
    pixelCountStats = modis.reduceRegion(reducer=ee.Reducer.mean(),geometry=b_box,bestEffort=True,maxPixels=1e9,scale=20)
    output = pixelCountStats.getInfo()
    if(not output):
        return 0 
    else:
        return output['Lai_500m']

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

    
def countNFIS(nfis_tif):
    tree_coverage = nfis_tif.where(np.isin(nfis_tif.data,[230,220,210,81])) #should forested wetland, 81, be included? 
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

def elevation(lat,lon,next_lat,next_lon):
    try:     
        ee.Initialize()
    except:
        ee.Authenticate()
        ee.Initialize()
    b_box = ee.Geometry.BBox(lon,lat,next_lon,next_lat)
    b_box_bounds = b_box.bounds()
    modis = ee.ImageCollection("JAXA/ALOS/AW3D30/V3_2").select('DSM').mean()
    modis = modis.clip(b_box_bounds)
    pixelCountStats = modis.reduceRegion(reducer=ee.Reducer.mean(),geometry=b_box,bestEffort=True,maxPixels=1e9,scale=20)
    return pixelCountStats.getInfo()['DSM']


# Define a method for displaying Earth Engine image tiles on a folium map.
def add_ee_layer(self, ee_object, vis_params, name) -> None:
    try:    
        # display ee.Image()
        if isinstance(ee_object, ee.image.Image):  
            map_id_dict = ee.Image(ee_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
            ).add_to(self)
        # display ee.ImageCollection()
        elif isinstance(ee_object, ee.imagecollection.ImageCollection):    
            print("ic here")
            ee_object_new = ee_object.mosaic()
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
            ).add_to(self)
        # display ee.Geometry()
        elif isinstance(ee_object, ee.geometry.Geometry):    
            folium.GeoJson(
            data = ee_object.getInfo(),
            name = name,
            overlay = True,
            control = True
        ).add_to(self)
        # display ee.FeatureCollection()
        elif isinstance(ee_object, ee.featurecollection.FeatureCollection):  
            ee_object_new = ee.Image().paint(ee_object, 0, 1)
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
        ).add_to(self)
    
    except:
        print("Could not display {}".format(name))