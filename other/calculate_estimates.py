from shapely.geometry import Polygon
from pyproj import Geod


#calculate 
from utils import getArea, getCoordinates
def aggregateCellValue(cell_value,lat,lon,next_lat,next_lon):
    area = getArea(lat,lon,next_lat,next_lon)
    # print(area)
    return area * cell_value




def getArea(lat,lon,next_lat,next_lon) -> float:
    #returns metre squared
    poly = Polygon([(lon,lat),(next_lon,lat),(lon,next_lat),(next_lon,next_lat),(lon,lat)])
    #put in counterclockwise rotation otherwise does not work
    geod = Geod(ellps="WGS84") #assume ellipsoid here
    #abs value, could be negative depending on orientation?
    # print(poly)
    area = geod.geometry_area_perimeter(poly)
    # print(area)
    area = abs(area[0])
    # print(area)
    return area
ordered_latitudes = []
ordered_longitudes = []
%env PROJECT_PATH=/Users/gclyne/thesis
%env NUM_CORES=4
import config
import csv
with open(f'{config.DATA_PATH}/grid_latitudes.csv',newline='') as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        ordered_latitudes.append(float(row[0]))

with open(f'{config.DATA_PATH}/grid_longitudes.csv',newline='') as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    for row in reader:
        ordered_longitudes.append(float(row[0]))

veg,soil,cwd,litter=0,0,0,0
count = 0
for row in eleven:
    lat,lon,next_lat,next_lon = getCoordinates((row[-2],row[-1]),ordered_latitudes,ordered_longitudes)
    veg += aggregateCellValue(row[0],lat,lon,next_lat,next_lon)
    soil += aggregateCellValue(row[1],lat,lon,next_lat,next_lon)
    litter += aggregateCellValue(row[2],lat,lon,next_lat,next_lon)
    cwd += aggregateCellValue(row[3],lat,lon,next_lat,next_lon)
    # print(veg,lat,lon,next_lat,next_lon,row)
    # count += 1
    # if count == 10:
    #     break

total = (veg + soil + litter + cwd) 