import multiprocessing
from other.utils import clipNFIS,getCoordinates
import rioxarray
import other.config as config
import numpy as np

class NFIS_Land_Cover():
    def __init__(self):
        self.columns =  ['no_change','water','snow_ice','rock_rubble','exposed_barren_land','bryoids','shrubs','wetland',
    'wetland-treed','herbs','coniferous','broadleaf','mixedwood','total_pixels','year','lat','lon']
        self.output_file_path = f'{config.DATA_PATH}/nfis_tree_cover_data.csv'
    
    
    def getRow(self,nfis_tif,year,lat,lon,next_lat,next_lon) -> list:
        """
        0 = no change
        20 = water
        31 = snow_ice
        32 = rock_rubble
        33 = exposed_barren_land
        40 = bryoids
        50 = shrubs
        80 = wetland
        81 = wetland-treed
        100 = herbs
        210 = coniferous
        220 = broadleaf
        230 = mixedwood
        """
        clipped_nfis = clipNFIS(nfis_tif,lat,lon,next_lat,next_lon)
        x = np.unique(clipped_nfis.data,return_counts=True)

        classes = [0,20,31,32,33,40,50,80,81,100,210,220,230]
        output_dict = {el:0 for el in classes}
        for index in range(len(x[0])):
            output_dict[x[0][index]] = x[1][index]
        #append year to row for timeseries potential,can drop this when doing testing - append lat lon for unique key (comibned w year)
        row = [*list(output_dict.values()),clipped_nfis.data.size,year,lat,lon]
        print(row)
        return row


    def generate_data(self,coordinates,ordered_latitudes,ordered_longitudes,writer,num_cores):
        for year in range(1984,1984+36,1):
            print(year)
            nfis_tif = rioxarray.open_rasterio(f'{config.NFIS_PATH}/CA_forest_VLCE2_{year}.tif',decode_coords='all',lock=False)
            x = iter(coordinates)
            p = multiprocessing.Pool(num_cores)
            with p:
                for _ in range(int(len(coordinates))):
                    lat,lon,next_lat,next_lon = getCoordinates(next(x),ordered_latitudes,ordered_longitudes)
                    #beware, failed child processes do not give error by default
                    p.apply_async(self.getRow,[nfis_tif,year,lat,lon,next_lat,next_lon],callback = writer.writerow)
                p.close()
                p.join()
            nfis_tif.close()

