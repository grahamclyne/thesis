import time
import config
import csv
import ee
from utils import getCoordinates,readCoordinates
from gee_utils import getMODISLAI, elevation
from argparse import ArgumentParser
# num of rows outputted should be # of years (31) * length of boreal coordinates (788) = 24428


def generateLaiData(writer,coordinates,start_time,year,latitudes,longitudes) -> None:
    iter_coords = iter(coordinates)
    for _ in range(int(len(coordinates))):
        lat,lon,next_lat,next_lon = getCoordinates(next(iter_coords),latitudes,longitudes)
        variable = getMODISLAI(lat,lon,next_lat,next_lon,year)
        writer.writerow([variable,lat,lon,year])
        print(f'{year},{lat},{lon} completed at {time.time() - start_time} seconds.')   


def generateElevationData(writer,coordinates,start_time,latitudes,longitudes) -> None:
    iter_coords = iter(coordinates)
    for _ in range(int(len(coordinates))):
        lat,lon,next_lat,next_lon = getCoordinates(next(iter_coords),latitudes,longitudes)
        variable = elevation(lat,lon,next_lat,next_lon)
        writer.writerow([variable,lat,lon])
        print(f'{lat},{lon} completed at {time.time() - start_time} seconds.')   


if __name__ == '__main__':
    try:     
        ee.Initialize()
    except:
        ee.Authenticate()
        ee.Initialize()
    start_time = time.time()

    parser = ArgumentParser()
    parser.add_argument('--output_file_name',  type=str)
    parser.add_argument('--variable_name',  type=str)
    parser.add_argument('--year',  type=str)

    args = parser.parse_args()

    managed_forest_coordinates = readCoordinates('managed_coordinates.csv',is_grid_file=False)
    ordered_latitudes = readCoordinates('grid_latitudes.csv',is_grid_file=True)
    ordered_longitudes = readCoordinates('grid_longitudes.csv',is_grid_file=True)

    file = open(f'{config.DATA_PATH}/{args.output_file_name}_{args.year}','w')
    writer = csv.writer(file)
    #write header row
    writer.writerow([args.variable_name,'year','lat','lon'])
    if(args.variable_name == 'lai'):
        generateLaiData(writer,managed_forest_coordinates,start_time,args.year,ordered_latitudes,ordered_longitudes)
    elif(args.variable_name == 'elev'):
        generateElevationData(writer,managed_forest_coordinates,start_time,ordered_latitudes,ordered_longitudes)
    file.close()


    

 

