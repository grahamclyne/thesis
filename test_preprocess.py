import rioxarray 
import pandas as pd
import numpy as np
import time
import geopandas as gpd


def preprocess():
    ecozones_coords = pd.read_csv('~/scratch/ecozones_coordinates.csv')
    ecozones_coords = ecozones_coords[ecozones_coords['zone'].isin(['Boreal Cordillera','Boreal PLain', 'Boreal Shield'])]
    # ecozones = gpd.read_file('data/shapefiles/ecozones.shp').to_crs('epsg:4326')
    # ecozones = ecozones.where(ecozones['ZONE_NAME'].isin(['Boreal Shield','Boreal Cordillera','Boreal PLain']))
    from preprocessing.utils import getGeometryBoxes
    boxes = getGeometryBoxes(ecozones_coords)
    print(len(boxes))
    forest_df = pd.DataFrame()
    for_har = rioxarray.open_rasterio('~/scratch/reprojected_4326_CA_Forest_Harvest_1985-2020_test.tif',chunks=True)
    # dissolved_ecozones = ecozones.dissolve()
    # gdf = gpd.GeoDataFrame(geometry=[dissolved_ecozones.geometry[0]])
    # for_har = for_har.rio.clip(gdf.geometry, ecozones.crs,drop=True)

    # for_har = for_har.rio.clip(ecozones.geometry,'epsg:4326',drop=True)
    last_year_cell = 0
    for year in range(1984,2001):
        print(year)
        start = time.time()

        cur_forest = rioxarray.open_rasterio(f'~/scratch/reprojected_4326_CA_forest_{year}.tif',chunks=True)
        # last_year_forest = rioxarray.open_rasterio(f'home/gclyne/scratch/reprojected_4326_CA_forest_{year}.tif',chunks=True)
        for box in boxes:
            print(box.bounds)
            lon = box.bounds[0]
            next_lon = box.bounds[2]
            lat = box.bounds[1]
            next_lat = box.bounds[3]
            cur_cell = cur_forest.sel(band=1,x=slice(lon,next_lon),y=slice(next_lat,lat))
            # last_year_cell = last_year_forest.sel(band=1,x=slice(lon,next_lon),y=slice(next_lat,lat))
            for_har_cell = for_har.sel(band=1,x=slice(lon,next_lon),y=slice(next_lat,lat))
            cur_cell = np.where(cur_cell.data > 200,1,0)
            # last_year_cell = np.where(last_year_cell.data > 200,1,0)
            for_har_prev_harvest = np.where(for_har_cell.data < year,1,0)
            harvested = np.where(for_har_cell.data == year,1,0)
            #needs to be previously harvested, and not considered forest in the previous year
            new_growth = np.where((cur_cell == 1) & (last_year_cell == 0) & (for_har_prev_harvest == 1),1,0)
            total_size = cur_cell.size
            percentage_growth = new_growth.sum() / total_size
            tree_cover = cur_cell.sum() / total_size
            percent_harvested = harvested.sum() / total_size
            forest_df = pd.concat([forest_df,pd.DataFrame({'lat':lat,'lon':lon,'year':year,'percentage_growth':percentage_growth.compute(),'tree_cover':tree_cover.compute(),'percent_harvested':percent_harvested.compute()}, index=[0])],ignore_index=True)
        last_year_cell = cur_cell.copy()
        end = time.time()
        print(end - start)
    forest_df.to_csv('forest_df.csv',index=False)

if __name__ == '__main__':
    start = time.time()
    preprocess()
    end = time.time()
    print(end - start)