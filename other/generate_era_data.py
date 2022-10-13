import multiprocessing
from utils import eraYearlyAverage,getCoordinates,readCoordinates
import csv 
from datetime import date
import xarray as xr
import config
import time
import argparse

def getRow(year,lat,lon,next_lat,next_lon,file_name):
    print(year,lat,lon,multiprocessing.current_process())
    era_temp = xr.open_dataset(f'{config.ERA_PATH}/{file_name}',engine='netcdf4')
    era_yearly_avg = eraYearlyAverage(era_temp,lat,lon,next_lat,next_lon,year)
    era_temp.close()
    #append year to row for timeseries potential,can drop this when doing testing - append lat lon for unique key (comibned w year)
    row = [era_yearly_avg,year,lat,lon]
    return row


# num of rows outputted should be # of years (31) * length of boreal coordinates (788) = 24428
if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--era_file_name',type=str)
    args = parser.parse_args()
    managed_forest_coordinates = readCoordinates('managed_coordinates.csv',is_grid_file=False)
    ordered_latitudes = readCoordinates('grid_latitudes.csv',is_grid_file=True)
    ordered_longitudes = readCoordinates('grid_longitudes.csv',is_grid_file=True)

    year = date(1984,1,1).year

    era_output = open(f'{config.DATA_PATH}/era_{args.era_file_name.split(".")[0]}_data.csv','w')
    writer = csv.writer(era_output)
    writer.writerow([args.era_file_name,'year','lat','lon'])

    for year in range(year,year+36,1):
        x = iter(managed_forest_coordinates)
        p = multiprocessing.Pool(5)
        with p:
            for i in range(int(len(managed_forest_coordinates))):
                lat,lon,next_lat,next_lon = getCoordinates(next(x),ordered_latitudes,ordered_longitudes)
                #beware, failed child processes do not give error by default
                p.apply_async(getRow,[year,lat,lon,next_lat,next_lon,args.era_file_name],callback = writer.writerow)                
            p.close()
            p.join()
        duration = time.time() - start_time
    era_output.close()
    duration = time.time() - start_time
    print(f'Completed in {duration} seconds.')
