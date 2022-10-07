import multiprocessing
from utils import eraYearlyAverage,getCoordinates,borealCoordinates,gridLatitudes,gridLongitudes
import csv 
from datetime import date
import xarray as xr
import config
import time
import argparse

def getRow(year,lat,lon,next_lat,next_lon,file_name):
    print(year,lat,lon,multiprocessing.current_process())
    era_temp = xr.open_dataset(f'{config.ERA_PATH}/{file_name}')
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
    boreal_coordinates = borealCoordinates()
    ordered_latitudes = gridLatitudes()
    ordered_longitudes = gridLongitudes()

    year = date(1984,1,1).year

    era_output = open(f'{config.DATA_PATH}/era_{args.era_file_name}_data.csv','w')
    writer = csv.writer(era_output)
    writer.writerow([args.era_file_name,'year','lat','lon'])

    for year in range(year,year+36,1):
        x = iter(boreal_coordinates)
        p = multiprocessing.Pool(5)
        with p:
            for i in range(int(len(boreal_coordinates))):
                lat,lon,next_lat,next_lon = getCoordinates(next(x),ordered_latitudes,ordered_longitudes)
                #beware, failed child processes do not give error by default
                p.apply_async(getRow,[year,lat,lon,next_lat,next_lon,args.era_file_name],callback = writer.writerow)
            p.close()
            p.join()
        duration = time.time() - start_time
    era_output.close()
    duration = time.time() - start_time
    print(f'Completed in {duration} seconds.')
