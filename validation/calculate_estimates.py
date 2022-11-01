import config
import csv
import numpy as np
from utils import getArea, getCoordinates
import pandas as pd
def aggregateCellValue(cell_value,lat,lon,next_lat,next_lon):
    area = getArea(lat,lon,next_lat,next_lon)
    # print(area)
    return area * cell_value

ordered_latitudes = []
ordered_longitudes = []


with open(f'{config.DATA_PATH}/grid_latitudes.csv',newline='') as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        ordered_latitudes.append(float(row[0]))

with open(f'{config.DATA_PATH}/grid_longitudes.csv',newline='') as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        ordered_longitudes.append(float(row[0]))

data = np.genfromtxt(f'/Users/gclyne/thesis/data/cesm_data.csv',delimiter=',',skip_header=1)
fifteen = data[data[:,-3] == 2015]
veg,soil,cwd,litter=0,0,0,0
count = 0
for row in fifteen:
    lat,lon,next_lat,next_lon = getCoordinates((row[-2],row[-1]),ordered_latitudes,ordered_longitudes)
    veg += aggregateCellValue(row[0],lat,lon,next_lat,next_lon)
    soil += aggregateCellValue(row[1],lat,lon,next_lat,next_lon)
    litter += aggregateCellValue(row[2],lat,lon,next_lat,next_lon)
    cwd += aggregateCellValue(row[3],lat,lon,next_lat,next_lon)

total = (veg + soil + litter + cwd) 
print('{:e}'.format(total))
print('{:e}'.format(veg))
print('{:e}'.format(soil))
print('{:e}'.format(litter))
print('{:e}'.format(cwd))