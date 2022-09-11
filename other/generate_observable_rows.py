import multiprocessing
from utils import clipNFIS,countNFIS,eraYearlyAverage,getMODISLAI,elevation,getCoordinates
import csv 
from datetime import date
import rioxarray
import xarray as xr
from constants import DATA_FOLDER
import ee 
import time
import multiprocessing

def getRow(nfis_tif,year,lat,lon,next_lat,next_lon) -> None:
    print(lat,lon,multiprocessing.current_process())
    # nfis_tif = rioxarray.open_rasterio(f'/Users/gclyne/Downloads/CA_forest_VLCE2_{year}/CA_forest_VLCE2_{year}.tif',decode_coords='all')
    era_temp = xr.open_dataset('/Users/gclyne/thesis/data/_2m_temperature.nc')

    clipped_nfis = clipNFIS(nfis_tif,lat,lon,next_lat,next_lon)
    observed_tree_cover = countNFIS(clipped_nfis)
    era_yearly_avg = eraYearlyAverage(era_temp,lat,lon,next_lat,next_lon,year)
    lai = getMODISLAI(lat,lon,next_lat,next_lon,year,0,{})
    elev = elevation(lat,lon,next_lat,next_lon,0,{})
    nfis_tif.close()
    era_temp.close()
    #append year to row for timeseries potential,can drop this when doing testing
    row = [observed_tree_cover,era_yearly_avg,lai,elev,year,lat,lon]
    print(row)
    return row


# num of rows outputted should be # of years (31) * length of boreal coordinates (788) = 24428
if __name__ == '__main__':
    start_time = time.time()

    boreal_coordinates = []
    ordered_latitudes = []
    ordered_longitudes = []

    with open('/Users/gclyne/thesis/boreal_latitudes_longitudes.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            boreal_coordinates.append((float(row[0]),float(row[1])))

    with open('grid_latitudes.csv',newline='') as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        for row in reader:
            ordered_latitudes.append(float(row[0]))

    with open('grid_longitudes.csv',newline='') as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        for row in reader:
            ordered_longitudes.append(float(row[0]))

    year = date(1984,1,1).year
    range(year, year + 31, 1)

    observable_rows = open(f'{DATA_FOLDER}observable_data.csv','w')
    writer = csv.writer(observable_rows)
    writer.writerow(['observed_tree_cover','era_temp2m','lai','elev','year','lat','lon'])
    lock = multiprocessing.Lock()
    try:     
        ee.Initialize()
    except:
        ee.Authenticate()
        ee.Initialize()
    p = multiprocessing.Pool(5)

    for year in range(year,year+31,1):
        print(year)
        #open NFIS file for year
        nfis_tif = rioxarray.open_rasterio(f'/Users/gclyne/Downloads/CA_forest_VLCE2_{year}/CA_forest_VLCE2_{year}.tif',decode_coords='all',lock=False)
        #open ERA file
        # era_temp = xr.open_dataset('/Users/gclyne/thesis/data/_2m_temperature.nc')
        x = iter(boreal_coordinates)
        p = multiprocessing.Pool(4)
        with p:
            for i in range(int(len(boreal_coordinates))):
                lat,lon,next_lat,next_lon = getCoordinates(next(x))
                p.apply_async(getRow,[nfis_tif,year,lat,lon,next_lat,next_lon],callback = writer.writerow)
                # lat,lon,next_lat,next_lon = getCoordinates(next(x))
                # x2 = p.apply_async(getRow,[nfis_tif,year,lat,lon,next_lat,next_lon])
                # lat,lon,next_lat,next_lon = getCoordinates(next(x))
                # x3 = p.apply_async(getRow,[nfis_tif,year,lat,lon,next_lat,next_lon])
                # lat,lon,next_lat,next_lon = getCoordinates(next(x))
                # x4 = p.apply_async(getRow,[nfis_tif,year,lat,lon,next_lat,next_lon])            
                # lat,lon,next_lat,next_lon = getCoordinates(next(x))
                # x5 = p.apply_async(getRow,[nfis_tif,year,lat,lon,next_lat,next_lon])
                # writer.writerow(x1.get())
                # writer.writerow(x2.get())
                # writer.writerow(x3.get())
                # writer.writerow(x4.get())
                # writer.writerow(x5.get())
            p.close()
            p.join()
        nfis_tif.close()
    observable_rows.close()
    duration = time.time() - start_time
    print(f'Completed in {duration} seconds.')