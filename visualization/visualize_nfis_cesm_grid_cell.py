import rioxarray
import numpy as np
import geopandas
from shapely.geometry import mapping 
import xarray 
from pyproj import Transformer, CRS
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from other.utils import scaleLongitudes


nfis_tif = rioxarray.open_rasterio('/Users/gclyne/Downloads/CA_forest_VLCE2_1984/CA_forest_VLCE2_1984.tif',decode_coords='all')
tree_cover_data = xarray.open_dataset('/Users/gclyne/thesis/data/treeFrac_Lmon_CESM2_land-hist_r1i1p1f1_gn_194901-201512.nc')
shape_file = geopandas.read_file('/Users/gclyne/thesis/data/NABoreal.shp',crs="epsg:4326")

tree_cover_data['lon'] = scaleLongitudes(tree_cover_data['lon'])
tree_cover_data = tree_cover_data['treeFrac']
tree_cover_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
tree_cover_data.rio.write_crs("epsg:4326", inplace=True)
clipped = tree_cover_data.rio.clip(shape_file.geometry.apply(mapping), shape_file.crs,drop=True)

lat = clipped.lat.values[15]
lon = clipped.lon.values[32]
next_lat = clipped.lat.values[16]
next_lon = clipped.lon.values[33]


#see readme from downloaded, the crs is epsg:3978 (thanks qgis!)
#custom_crs = pyproj.crs.CRS('+proj=lcc +lat_0=49 +lon_0=-95 +lat_1=49 +lat_2=77 +x_0=0 +y_0=0 +datum=NAD83 +units=m +no_defs +type=crs')
epsg_4326 = CRS.from_epsg(4326)


transformer =Transformer.from_crs(epsg_4326,'epsg:3978')
x1,y1 = transformer.transform(lat,lon)
x2,y2 = transformer.transform(next_lat,next_lon)



#this is a hack because need to get conical coordinates (at least smaller size) before reproejcting to 4326, otherwise takes too long
bounds = 50000
sl = nfis_tif.isel(
    x=(nfis_tif.x >= x1-bounds) & (nfis_tif.x < x2+bounds),
    y=(nfis_tif.y >= y1-bounds) & (nfis_tif.y < y2+bounds),
    band=0
    )

sl = sl.rio.reproject('EPSG:4236')

sl = sl.isel(
    x=np.logical_and(sl.x >= lon, sl.x <= next_lon),
    y=np.logical_and(sl.y >= lat, sl.y <= next_lat)
    )

clipped = clipped.isel(
    lon=np.logical_and(clipped.lon >= lon, clipped.lon <= next_lon),
    lat=np.logical_and(clipped.lat >= lat, clipped.lat <= next_lat),
    )


#export images to raster
# sl.rio.to_raster('test.tif')
# clipped.rio.to_raster('clipped_full.tif')


map = Basemap(projection='merc',llcrnrlon=lon-1,llcrnrlat=lat-1,urcrnrlon=next_lon+1,urcrnrlat=next_lat+1,resolution='h') # projection, lat/lon extents and resolution of polygons to draw

#the two images should overlay each other, and only show the nfis image

#plot cesm grid
lons,lats= np.meshgrid(clipped.lon,clipped.lat) 
x,y = map(lons,lats)
map.pcolormesh(x,y,clipped[10,0:1,0:1]) # get only the one grid cell

#plot nfis grid
lons,lats= np.meshgrid(sl.x,sl.y)
x,y = map(lons,lats)
map.pcolormesh(x,y,sl)




map.drawcoastlines()
map.drawcountries()
map.drawlsmask(land_color='Linen', ocean_color='#CCFFFF') # can use HTML names or codes for colors

parallels = np.arange(30,90,5.) # make latitude lines ever 5 degrees
meridians = np.arange(-140,-80,5.) # make longitude lines every 5 degrees
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
plt.savefig('grid_cell_plot.png')