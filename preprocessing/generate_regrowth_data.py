from utils import clipNFIS,getCoordinates
import csv 
import rioxarray
import config
import numpy as np 


class ObservableDataset():
    def __init__(self,columns,output_file_path,getRow,generate):
        self.columns =  columns
        self.output_file_path = output_file_path
        self.getRow = getRow
        self.generate = generate

def getRow(nfis_tif:rioxarray.Dataset,lat:float,lon:float,next_lat:float,next_lon:float) -> list:
        clipped_nfis = clipNFIS(nfis_tif,lat,lon,next_lat,next_lon)
        x = np.unique(clipped_nfis.data,return_counts=True)
        classes = [0] + [x for x in range(1,33,1)]
        output_dict = {el:0 for el in classes}
        for index in range(len(x[0])):
            output_dict[x[0][index]] = x[1][index]
        row = [*list(output_dict.values()),clipped_nfis.data.size,lat,lon]
        print(row)
        return row

def generate(coordinates:list,ordered_latitudes:list,ordered_longitudes:list,writer:csv.writer,num_cores:int,cfg) -> None:
    nfis_tif = rioxarray.open_rasterio(f'data/CA_forest_harvest_years2recovery/CA_forest_harvest_years2recovery.tif',decode_coords='all',lock=False)
    x = iter(coordinates)
    for _ in range(int(len(coordinates))):
        lat,lon = next(x)
        next_lat,next_lon = getCoordinates(lat,lon,cfg)
        print(lat,lon)
        writer.writerow(getRow(nfis_tif,lat,lon,next_lat,next_lon))
    nfis_tif.close()

NFIS_Regrowth = ObservableDataset(
    columns = [x for x in range(1985,2015)] + ['total_pixels','year','lat','lon'],
    output_file_path=f'{config.DATA_PATH}/nfis_regrowth_data.csv',
    getRow = getRow,
    generate = generate
)