import multiprocessing
from preprocessing.utils import clipNFIS,getCoordinates
import rioxarray
import numpy as np

class NFIS_Land_Cover():
    def __init__(self,cfg):
        self.columns =  ['no_change','water','snow_ice','rock_rubble','exposed_barren_land','bryoids','shrubs','wetland',
    'wetland-treed','herbs','coniferous','broadleaf','mixedwood','total_pixels','year','lat','lon']
        self.cfg = cfg
        self.output_file_path = f'{cfg.data}/nfis_tree_cover_data_ecozones.csv'
    
    
    def getRow(self,nfis_tif,year,lat,lon,next_lat,next_lon) -> list:
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


    def generate_data(self,coordinates,writer,num_cores):
        for year in range(1984,1984+36,1):
            print(year)
            if(self.cfg.data == '/Users/gclyne/thesis/data'):
                nfis_tif = rioxarray.open_rasterio(f'{self.cfg.data}/NFIS/CA_forest_VLCE2_{year}/CA_forest_VLCE2_{year}.tif',lock=False)
            else:
                nfis_tif = rioxarray.open_rasterio(f'{self.cfg.data}/CA_forest_VLCE2_{year}.tif',decode_coords='all',lock=False)
            x = iter(coordinates)
            p = multiprocessing.Pool(num_cores)
            with p:
                for _ in range(int(len(coordinates))):
                    lat,lon = next(x)
                    next_lat,next_lon = getCoordinates(lat,lon,self.cfg)
                    #beware, failed child processes do not give error by default
                    p.apply_async(self.getRow,[nfis_tif,year,lat,lon,next_lat,next_lon],callback = writer.writerow)
                    # x.get()
                p.close()
                p.join()
            nfis_tif.close()

