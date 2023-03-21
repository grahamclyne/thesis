import geopandas
import xarray as xr
import csv
import preprocessing.config as config
from preprocessing.utils import scaleLongitudes,clipWithShapeFile
import hydra
from omegaconf import DictConfig

#to run: python -m preprocessing.generate_shapefile_lats_lons
#to run: python -m preprocessing.generate_shapefile_lats_lons 



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    #this could be any CESM CMIP6 file ?

    tree_cover_data = xr.open_dataset(f'{cfg.data}/CESM/treeFrac_Lmon_CESM2_historical_r11i1p1f1_gn_199901-201412.nc')

    shape_file = geopandas.read_file(f'{cfg.data}/shapefiles/{cfg.study_area}.shp',crs="epsg:4326")
    lons = open(f'{cfg.data}/grid_longitudes.csv','w')
    lats = open(f'{cfg.data}/grid_latitudes.csv','w')
    lon_writer = csv.writer(lons)
    lat_writer = csv.writer(lats)
    tree_cover_data = scaleLongitudes(tree_cover_data)
    #important step - in case the tree cover data has any NaN values
    tree_cover_data = tree_cover_data.fillna(0)
    # tree_cover_data = tree_cover_data['treeFrac'].isel(time=0)
    # tree_cover_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    # tree_cover_data.rio.write_crs("epsg:4326", inplace=True)
    # clipped = tree_cover_data.rio.clip(shape_file.geometry.apply(lambda x: x.__geo_interface__), shape_file.crs, drop=True, invert=False, all_touched=False, from_disk=False)
    clipped = clipWithShapeFile(tree_cover_data,'treeFrac',shape_file)

    #generate csv of total lats and lons for whole world grid
    for lon in tree_cover_data.lon:
        lon_writer.writerow([lon.values])
    for lat in tree_cover_data.lat:
        lat_writer.writerow([lat.values])

    clipped = clipped.groupby('time.year').mean()
    print(clipped)
    grouped = clipped.to_dataframe().reset_index()
    # grouped = grouped.to_dataframe()
    print(grouped)
    lat_lon = grouped[grouped.year == 1999].groupby(['lat','lon']).mean().reset_index().dropna()[['lat','lon']].drop_duplicates()
    print(lat_lon)
    lat_lon.to_csv(f'{cfg.data}/{cfg.study_area}_coordinates.csv',index=False)


    lons.close()
    lats.close()

main()