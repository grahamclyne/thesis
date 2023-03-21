import multiprocessing
from preprocessing.utils import clipNFIS,getCoordinates
import preprocessing.config as config
import multiprocessing
import rioxarray

class NFIS_Biomass():
    def __init__(self,cfg):
        self.columns =  ['agb','lat','lon']
        self.cfg = cfg
        self.output_file_path = f'{cfg.data}/nfis_agb_ecozones.csv'
    
    # @property
    # def output_file_path(self):
    #     return self._output_file_path
    
    def getRow(nfis_tif,lat,lon,next_lat,next_lon):
        print(lat,lon,multiprocessing.current_process())
        agb = clipNFIS(nfis_tif,lat,lon,next_lat,next_lon).mean().values
        row = [agb,lat,lon]
        print(row)
        return row

    def generate_data(self,coordinates,writer,num_cores):
        nfis_tif = rioxarray.open_rasterio(f'{self.cfg.data}/CA_forest_total_biomass_2015_NN/CA_forest_total_biomass_2015.tif',lock=False)
        x = iter(coordinates)
        p = multiprocessing.Pool(num_cores)
        with p:
            for _ in range(int(len(coordinates))):
                lat,lon = next(x)
                next_lat,next_lon = getCoordinates(lat,lon,self.cfg)
                # print(lat,lon)
                # row = getRow(nfis_tif,lat,lon,next_lat,next_lon)
                # writer.writerow(row)
                #beware, failed child processes do not give error by default
                x = p.apply_async(self.getRow,[nfis_tif,lat,lon,next_lat,next_lon],callback = writer.writerow)
                # x.get() use this line for debugging
            p.close()
            p.join()