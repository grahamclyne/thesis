import time
import config
import csv
from datetime import date
import ee
from utils import getCoordinates, getMODISLAI,elevation
from argparse import ArgumentParser
# num of rows outputted should be # of years (31) * length of boreal coordinates (788) = 24428


def generateLaiData(writer,boreal_coordinates,start_time,year,latitudes,longitudes) -> None:
    for year in range(year,year+35,1):
        boreal_iter = iter(boreal_coordinates)
        for i in range(int(len(boreal_coordinates))):
            lat,lon,next_lat,next_lon = getCoordinates(next(boreal_iter),latitudes,longitudes)
            variable = getMODISLAI(lat,lon,next_lat,next_lon,year)
            writer.writerow([variable,lat,lon,year])
            print(f'{year},{lat},{lon} completed at {time.time() - start_time} seconds.')   


def generateElevationData(writer,boreal_coordinates,start_time,latitudes,longitudes) -> None:
    boreal_iter = iter(boreal_coordinates)
    for i in range(int(len(boreal_coordinates))):
        lat,lon,next_lat,next_lon = getCoordinates(next(boreal_iter),latitudes,longitudes)
        variable = elevation(lat,lon,next_lat,next_lon)
        writer.writerow([variable,lat,lon])
        print(f'{year},{lat},{lon} completed at {time.time() - start_time} seconds.')   


if __name__ == '__main__':
    try:     
        ee.Initialize()
    except:
        ee.Authenticate()
        ee.Initialize()
    start_time = time.time()

    parser = ArgumentParser()
    parser.add_argument('--file_name',  type=str)
    parser.add_argument('--variable_name',  type=str)
    args = parser.parse_args()

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
    file = open(f'{config.DATA_PATH}/{args.file_name}','w')
    writer = csv.writer(file)
    writer.writerow([args.variable_name,'year','lat','lon'])
    if(args.variable_name == 'lai'):
        generateLaiData(writer,boreal_coordinates,start_time,year,ordered_latitudes,ordered_longitudes)
    elif(args.variable_name == 'elev'):
        generateElevationData(writer,boreal_coordinates,start_time,ordered_latitudes,ordered_longitudes)
    file.close()


    

 

