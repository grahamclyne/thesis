import rioxarray 
import pandas as pd
import numpy as np
import time
import geopandas as gpd


def preprocess():
    ecozones_coords = pd.read_csv('~/scratch/ecozones_coordinates.csv')
    ecozones_coords = ecozones_coords[ecozones_coords['zone'].isin(['Boreal Cordillera','Boreal PLain', 'Boreal Shield'])]
    from preprocessing.utils import getGeometryBoxes
    boxes = getGeometryBoxes(ecozones_coords)
    forest_df = pd.DataFrame()
    agb = rioxarray.open_rasterio('~/scratch/reprojected_4326_CA_forest_total_biomass_2015.tif')
    for box in boxes:
        lon = box.bounds[0]
        next_lon = box.bounds[2]
        lat = box.bounds[1]
        next_lat = box.bounds[3]
        agb_cell = agb.sel(band=1,x=slice(lon,next_lon),y=slice(next_lat,lat))
        total = agb_cell.mean()
        forest_df = pd.concat([forest_df,pd.DataFrame({'lat':lat,'lon':lon,'agb':total}, index=[0])],ignore_index=True)
        end = time.time()
        print(end - start)
    forest_df.to_csv('nfis_agb_df.csv',index=False)

if __name__ == '__main__':
    start = time.time()
    preprocess()
    end = time.time()
    print(end - start)