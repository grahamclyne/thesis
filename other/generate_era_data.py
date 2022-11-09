import multiprocessing
from other.utils import getCoordinates
import xarray as xr
import other.config as config
import cdsapi
import numpy as np 

class ERA_Dataset():
    def __init__(self,variable):
        self.variable = variable
        self.columns = [f'{variable}','year','lat','lon']
        self.output_file_path = f'{config.GENERATED_DATA}/era_{variable}_data.csv'


    def download_era(self,variable):
        c = cdsapi.Client()



        c.retrieve(
            'reanalysis-era5-land-monthly-means',
            {
                'format': 'netcdf',
                'product_type': 'monthly_averaged_reanalysis',
                'variable': variable,
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                ],
                'time': '00:00',
                'area': [
                    80, -140, 40,
                    -40,
                ],
                'year': [
                    '1984', '1985', '1986',
                    '1987', '1988', '1989',
                    '1990', '1991', '1992',
                    '1993', '1994', '1995',
                    '1996', '1997', '1998',
                    '1999', '2000', '2001',
                    '2002', '2003', '2004',
                    '2005', '2006', '2007',
                    '2008', '2009', '2010',
                    '2011', '2012', '2013',
                    '2014', '2015','2016','2017','2018','2019'
                ],
            },
            f'{config.ERA_PATH}/{variable}.nc')

    def eraYearlyAverage(self,era,lat,lon,next_lat,next_lon,year):
        era = era.groupby('time.year').mean()
        era = era.isel(
        year=year - era.year.min().values.item(), #this only works if first year of data is 1984
        latitude=np.logical_and(era.latitude >= lat,era.latitude <= next_lat), 
        longitude=np.logical_and(era.longitude >= lon, era.longitude <= next_lon)
        )
        return (era.sum() / (len(era.latitude) * len(era.longitude)))[list(era.keys())[0]].values.item()


    def getRow(self,year,lat,lon,next_lat,next_lon,file_name):
        print(year,lat,lon,multiprocessing.current_process())
        era_temp = xr.open_dataset(f'{config.ERA_PATH}/{file_name}.nc',engine='netcdf4')
        era_yearly_avg = self.eraYearlyAverage(era_temp,lat,lon,next_lat,next_lon,year)
        era_temp.close()
        #append year to row for timeseries potential,can drop this when doing testing - append lat lon for unique key (comibned w year)
        row = [era_yearly_avg,year,lat,lon]
        return row

    def generate_data(self,managed_forest_coordinates,ordered_latitudes,ordered_longitudes,writer,num_cores):
        self.download_era(self.variable)
        for year in range(1984,2020,1):
                x = iter(managed_forest_coordinates)
                p = multiprocessing.Pool(num_cores)
                with p:
                    for _ in range(int(len(managed_forest_coordinates))):
                        lat,lon,next_lat,next_lon = getCoordinates(next(x),ordered_latitudes,ordered_longitudes)
                        #beware, failed child processes do not give error by default
                        p.apply_async(self.getRow,[year,lat,lon,next_lat,next_lon,self.variable + '.nc'],callback = writer.writerow)                
                    p.close()
                    p.join()