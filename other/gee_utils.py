import ee
import folium

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