import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
import csv
file = '/home/graham/code/thesis/data/treeFrac_Lmon_CESM2_land-hist_r1i1p1f1_gn_194901-201512.nc'
dset = xr.open_dataset(file)


boreal = gpd.read_file('/home/graham/Downloads/boreal/NABoreal.shp')
canada = gpd.read_file('/home/graham/Downloads/gpr_000b11a_e.shp')
# convert projection to familiar lat/lon
boreal = boreal.to_crs('epsg:4326')
canada = canada.to_crs('epsg:4326')

output = open('canadian_boreal_coordinates.csv', 'w')
csv_writer = csv.writer(output)
for lat in dset['lat']:
    for lon in dset['lon']:
        lon = lon.values - 180
        print(lat.values,lon)
        if(lat.values < 40 or lat.values > 80 or lon < -170 or lon > -40):
            continue
        point = Point(lon,lat.values)
        in_canada = False
        for polygon in canada['geometry']:
            if(point.within(polygon)):
                in_canada = True
                break
        if(in_canada):
            for polygon in boreal['geometry']:
                if(point.within(polygon)):
                    csv_writer.writerow([lat.values,lon])
                    break
output.close()