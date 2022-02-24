import folium
import geopandas as gpd
from sentinelsat.sentinel import SentinelAPI
import rasterio 
import matplotlib.pyplot as plt
from rasterio import plot
from rasterio.plot import show
from rasterio.mask import mask
from osgeo import gdal

m = folium.Map([53.5461, -113.4938], zoom_start=11)
boundary = gpd.read_file(r'map.geojson')
folium.GeoJson(boundary).add_to(m)
m.save('map.html')

footprint = None
for i in boundary['geometry']:
    footprint = i
    
user = 'gclyne'
password = 'fib12358'
api = SentinelAPI(user, password, 'https://scihub.copernicus.eu/dhus')
products = api.query(footprint,
                     date = ('20210109', '20210510'),
                     platformname = 'Sentinel-2',
                    #  processinglevel = 'Level-2A',
                     cloudcoverpercentage = (0, 10))

gdf = api.to_geodataframe(products)
gdf_sorted = gdf.sort_values(['cloudcoverpercentage'], ascending=[True])

# import matplotlib.pyplot as plt
# gdf_sorted.plot(column = 'uuid', cmap=None)

# plt.savefig('world.jpg')




api.download(gdf_sorted['uuid'][0])

# bands = r'...\GRANULE\L2A_T18TWL_A025934_20200609T155403\IMG_DATA\R10m'
# print(bands)
# blue = rasterio.open(bands+'\T18TWL_20200609T154911_B02_10m.jp2') 
# green = rasterio.open(bands+'\T18TWL_20200609T154911_B03_10m.jp2') 
# red = rasterio.open(bands+'\T18TWL_20200609T154911_B04_10m.jp2') 
# with rasterio.open('image_name.tiff','w',driver='Gtiff', width=blue.width, height=blue.height, count=3, crs=blue.crs,transform=blue.transform, dtype=blue.dtypes[0]) as rgb:
#     rgb.write(blue.read(1),3) 
#     rgb.write(green.read(1),2) 
#     rgb.write(red.read(1),1) 
#     rgb.close()