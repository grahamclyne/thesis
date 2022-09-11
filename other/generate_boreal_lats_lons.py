import rioxarray
import numpy as np
import geopandas
from shapely.geometry import mapping 
import xarray as xr
import csv

#this could be any CESM CMIP6 file 
tree_cover_data = xr.open_dataset('/Users/gclyne/thesis/data/treeFrac_Lmon_CESM2_land-hist_r1i1p1f1_gn_194901-201512.nc')

shape_file = geopandas.read_file('/Users/gclyne/thesis/data/NABoreal.shp',crs="epsg:4326")
canada_shape_file = geopandas.read_file('/Users/gclyne/Downloads/lpr_000b21a_e/lpr_000b21a_e.shp')
f = open('boreal_latitudes_longitudes.csv', 'w')
lons = open('grid_longitudes.csv','w')
lats = open('grid_latitudes.csv','w')
writer = csv.writer(f)
lon_writer = csv.writer(lons)
lat_writer = csv.writer(lats)
# canada_shape_file = canada_shape_file.to_crs('epsg:4326')
tree_cover_data['lon'] = tree_cover_data['lon'] - 360 if np.any(tree_cover_data['lon'] > 180) else tree_cover_data['lon']
tree_cover_data = tree_cover_data['treeFrac']
tree_cover_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
tree_cover_data.rio.write_crs("epsg:4326", inplace=True)
clipped = tree_cover_data.rio.clip(shape_file.geometry.apply(mapping), shape_file.crs,drop=True)
clipped = clipped.rio.clip(canada_shape_file.geometry.apply(mapping),canada_shape_file.crs,drop=True)
#select first time, doesnt matter which really
clipped = clipped.isel(time=0)

#generate csv of total lats and lons
for lon in tree_cover_data.lon:
    lon_writer.writerow([lon.values])
for lat in tree_cover_data.lat:
    lat_writer.writerow([lat.values])


#find non-null values, these are the boundaries for the boreal forest
for lat in clipped['lat']:
    for lon in clipped['lon']:
        # print(lat.values)
        x = clipped.sel(lat=lat.values,lon=lon.values)
        if(not np.isnan(x.values)):
            writer.writerow((lat.values,lon.values))




f.close()
lons.close()
lats.close()