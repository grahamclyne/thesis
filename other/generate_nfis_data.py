import multiprocessing
from utils import clipNFIS,countNFIS,eraYearlyAverage,getCoordinates
import csv 
from datetime import date
import rioxarray
import xarray as xr
import config
import time

def getRow(nfis_tif,year,lat,lon,next_lat,next_lon):
    print(lat,lon,multiprocessing.current_process())
    clipped_nfis = clipNFIS(nfis_tif,lat,lon,next_lat,next_lon)
    # observed_tree_cover = countNFIS(clipped_nfis)
    observed_tree_cover = countNFIS(clipped_nfis,[210,220,230,81])
    observed_wetland = countNFIS(clipped_nfis,[80])
    observed_shrub_bryoid_herb = countNFIS(clipped_nfis,[100,50,40])
    

    #append year to row for timeseries potential,can drop this when doing testing - append lat lon for unique key (comibned w year)
    row = [observed_tree_cover,year,lat,lon]
    return row


# num of rows outputted should be # of years (31) * length of boreal coordinates (788) = 24428
if __name__ == '__main__':
    start_time = time.time()

    boreal_coordinates = []
    ordered_latitudes = []
    ordered_longitudes = []

    with open(f'{config.DATA_PATH}/boreal_latitudes_longitudes.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            boreal_coordinates.append((float(row[0]),float(row[1])))

    with open(f'{config.DATA_PATH}/grid_latitudes.csv',newline='') as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        for row in reader:
            ordered_latitudes.append(float(row[0]))

    with open(f'{config.DATA_PATH}/grid_longitudes.csv',newline='') as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        for row in reader:
            ordered_longitudes.append(float(row[0]))

    year = date(1984,1,1).year

    observable_rows = open(f'{config.DATA_PATH}/nfis_tree_cover_data.csv','w')
    writer = csv.writer(observable_rows)
    writer.writerow(['observed_tree_cover','year','lat','lon'])

    for year in range(year,year+36,1):
        print(year)
        nfis_tif = rioxarray.open_rasterio(f'{config.NFIS_PATH}/CA_forest_VLCE2_{year}/CA_forest_VLCE2_{year}.tif',decode_coords='all',lock=False)
        x = iter(boreal_coordinates)
        p = multiprocessing.Pool(5)
        with p:
            for i in range(int(len(boreal_coordinates))):
                lat,lon,next_lat,next_lon = getCoordinates(next(x),ordered_latitudes,ordered_longitudes)
                #beware, failed child processes do not give error by default
                p.apply_async(getRow,[nfis_tif,year,lat,lon,next_lat,next_lon],callback = writer.writerow)
            p.close()
            p.join()
        duration = time.time() - start_time
        print(f'{year} completed at {duration} seconds.')    
        nfis_tif.close()
    observable_rows.close()
    duration = time.time() - start_time
    print(f'Completed in {duration} seconds.')
