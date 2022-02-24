import geopandas as gpd

# load zips with the source projection
shapefilename = '/home/graham/code/thesis/boreal/NABoreal.shp'
zips = gpd.read_file(shapefilename)
# convert projection to familiar lat/lon
zips = zips.to_crs('epsg:4326')

import xarray as xr
file = '/home/graham/code/thesis/data/treeFrac_Lmon_CESM2_land-hist_r1i1p1f1_gn_194901-201512.nc'
dset = xr.open_dataset(file)

from shapely.geometry import Point
for lat in dset['lat']:
    for lon in dset['lon']:
        lon = lon.values - 180
        if(lat.values < 40 or lat.values > 70 or lon < -160 or lon > -40):
            continue
        point = Point(lon,lat.values)
        for polygon in zips['geometry']:
            if(point.within(polygon)):
               print(point) 

from shapely.ops import unary_union

union = unary_union(zips['geometry'])
print(union.area)
print(type(union))