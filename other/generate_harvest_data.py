import multiprocessing
from utils import clipNFIS,countNFIS,getCoordinates,readCoordinates
import csv 
from datetime import date
import rioxarray
import config
import time

def getRow(nfis_tif,year,lat,lon,next_lat,next_lon):
    print(lat,lon,multiprocessing.current_process())
    clipped_nfis = clipNFIS(nfis_tif,lat,lon,next_lat,next_lon)
    observed_tree_cover = countNFIS(clipped_nfis,[year - 1900])
    #append year to row for timeseries potential,can drop this when doing testing - append lat lon for unique key (comibned w year)
    row = [observed_tree_cover,year,lat,lon]
    return row


# num of rows outputted should be # of years (31) * length of boreal coordinates (788) = 24428
if __name__ == '__main__':
    start_time = time.time()

    managed_forest_coordinates = readCoordinates('managed_coordinates.csv',is_grid_file=False)
    ordered_latitudes = readCoordinates('grid_latitudes.csv',is_grid_file=True)
    ordered_longitudes = readCoordinates('grid_longitudes.csv',is_grid_file=True)

    year = date(1985,1,1).year

    observable_rows = open(f'{config.DATA_PATH}/nfis_harvest_data.csv','w')
    writer = csv.writer(observable_rows)
    writer.writerow(['harvested_percentage','year','lat','lon'])

    for year in range(year,year+31,1):
        print(year)
        nfis_tif = rioxarray.open_rasterio(f'{config.NFIS_PATH}/CA_forest_harvest_mask_year_1985_2015/CA_harvest_year_1985_2015.tif',decode_coords='all',lock=False)
        x = iter(managed_forest_coordinates)
        p = multiprocessing.Pool(5)
        with p:
            for i in range(int(len(managed_forest_coordinates))):
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
