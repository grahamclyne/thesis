import geopandas
from shapely.geometry import mapping 
import xarray as xr
import csv
import preprocessing.config as config
import argparse
import numpy as np
from preprocessing.utils import scaleLongitudes 

#to run: python -m other.generate_shapefile_lats_lons --shape_file_path /Users/gclyne/thesis/data/shapefiles/NIR2016_MF.shp --output_file_path /Users/gclyne/thesis/data/managed_real_output.csv

#this could be any CESM CMIP6 file 
parser = argparse.ArgumentParser()
parser.add_argument('--shape_file_path',type=str)
parser.add_argument('--output_file_path',type=str)
args = parser.parse_args()
tree_cover_data = xr.open_dataset(f'{config.CESM_PATH}/treeFrac_Lmon_CESM2_land-hist_r1i1p1f1_gn_194901-201512.nc')

shape_file = geopandas.read_file(f'{args.shape_file_path}',crs="epsg:4326")
f = open(f'{args.output_file_path}', 'w')
lons = open(f'{config.DATA_PATH}/grid_longitudes.csv','w')
lats = open(f'{config.DATA_PATH}/grid_latitudes.csv','w')
writer = csv.writer(f)
lon_writer = csv.writer(lons)
lat_writer = csv.writer(lats)
tree_cover_data = scaleLongitudes(tree_cover_data,'lon')
tree_cover_data = tree_cover_data['treeFrac'].isel(time=0)
tree_cover_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
tree_cover_data.rio.write_crs("epsg:4326", inplace=True)
clipped = tree_cover_data.rio.clip([shape_file.geometry.apply(mapping)[0]], shape_file.crs,drop=True) #geometry needs to be in a list, [0] index for geometry is managed land

#generate csv of total lats and lons
for lon in tree_cover_data.lon:
    lon_writer.writerow([lon.values])
for lat in tree_cover_data.lat:
    lat_writer.writerow([lat.values])


#find non-null values, these are the boundaries for the forest
for lat in clipped['lat']:
    for lon in clipped['lon']:
        # print(lat.values)
        x = clipped.sel(lat=lat.values,lon=lon.values)
        if(not np.isnan(x.values)):
            writer.writerow((lat.values,lon.values))




f.close()
lons.close()
lats.close()