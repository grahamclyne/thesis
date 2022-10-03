import multiprocessing
import csv 
from utils import clipNFIS,getCoordinates
import config
import time
import multiprocessing
import rioxarray


def getRow(nfis_tif,lat,lon,next_lat,next_lon):
    print(lat,lon,multiprocessing.current_process())
    agb = clipNFIS(nfis_tif,lat,lon,next_lat,next_lon).mean().values
    return [agb,lat,lon]

if __name__ == "__main__":
    nfis_tif = rioxarray.open_rasterio(f'{config.NFIS_PATH}/CA_forest_total_biomass_2015_NN/CA_forest_total_biomass_2015.tif',lock=False)
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


    observable_rows = open(f'{config.DATA_PATH}/nfis_agb.csv','w')
    writer = csv.writer(observable_rows)
    writer.writerow(['agb','lat','lon'])

    # print(boreal_coor)
    x = iter(boreal_coordinates)
    # p = multiprocessing.Pool(5)
    # with p:
    for i in range(int(len(boreal_coordinates))):
        lat,lon,next_lat,next_lon = getCoordinates(next(x),ordered_latitudes,ordered_longitudes)
        print(lat,lon)
        row = getRow(nfis_tif,lat,lon,next_lat,next_lon)
        writer.writerow(row)
        #beware, failed child processes do not give error by default
        # p.apply_async(getRow,[nfis_tif,lat,lon,next_lat,next_lon],callback = writer.writerow)
        # x.get() use this line for debugging
        # p.close()
        # p.join()
    nfis_tif.close()
    observable_rows.close()
    duration = time.time() - start_time
    print(f'Completed in {duration} seconds.')
