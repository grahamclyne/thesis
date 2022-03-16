import xarray as xr
import cfgrib
import numpy as np
import cftime
# need to export ECCODES_DIR='/home/graham/eccodes'
# era = xr.load_dataset('/home/graham/Downloads/era_temp_precip_1984_2015.grib',filter_by_keys={'stepType': 'avgid'})
# x = era.groupby('time.year').mean('time')
def clipData(dataset,isERA):
    if(isERA):
        dataset = dataset.isel(latitude=np.logical_and(dataset.latitude<80,dataset.latitude > 40))
        dataset = dataset.isel(longitude=np.logical_and(dataset.longitude>-170,dataset.longitude <-50))
        dataset = dataset.isel(time=dataset.time >= np.datetime64('1984-01-01'))    

    else:
        dataset = dataset.isel(lat=np.logical_and(dataset.lat<80,dataset.lat > 40))
        dataset = dataset.isel(lon=np.logical_and(dataset.lon-360>-170,dataset.lon-360 <-50))   
        dataset = dataset.isel(time=dataset.time >= cftime.DatetimeNoLeap(1984, 1, 1))  
        # tp = tp.groupby('time.month').mean('time')
    # tp.to_netcdf('tp_groupedby_month.nc')
    # t2m = t2m_era_1984_2015.groupby('time.month').mean('time')
    # t2m.to_netcdf('t2m_groupedby_month.nc')
    return dataset.groupby('time').mean(...)    
import os
os.system('export ECCODES_DIR=/home/graham/eccodes')
# x.to_netcdf('era_groupedby_year.nc')
tp_era_1984_2001 = xr.open_dataset('/home/graham/Downloads/era_temp_precip_1984_2015.grib',filter_by_keys={'stepType': 'avgas'})
tp_era_2002_2015 = xr.open_dataset('/home/graham/Downloads/era_temp_precip_1984_2015.grib',filter_by_keys={'stepType': 'avgas'})
t2m_era_1984_2015 = xr.open_dataset('/home/graham/Downloads/era_temp_precip_1984_2015.grib',filter_by_keys={'stepType': 'avgid'})
# print(tp_era_1984_2001['tp'])
# print(type(tp_era_2002_2015))
tp = xr.merge([tp_era_2002_2015,tp_era_1984_2001])
data_folder = 'data/'
tree_cover_data = xr.open_dataset(data_folder + 'treeFrac_Lmon_CESM2_land-hist_r1i1p1f1_gn_194901-201512.nc')
cVeg_data = xr.open_dataset(data_folder + 'cVeg_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc')
cesm_temp_data = xr.open_dataset(data_folder + 'tas_Amon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc')
cesm_precip_data = xr.open_dataset(data_folder + 'pr_Amon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc')
print(t2m_era_1984_2015)
print(t2m_era_1984_2015.time[0])
clipData(tp,True).to_netcdf('era_boreal_monthly_tp.nc')
clipData(t2m_era_1984_2015,True).to_netcdf('era_boreal_monthly_t2m.nc')
clipData(tree_cover_data,False).to_netcdf('cesm_boreal_monthly_treeFrac.nc')
clipData(cVeg_data,False).to_netcdf('cesm_boreal_monthly_cVeg.nc')
clipData(cesm_temp_data,False).to_netcdf('cesm_boreal_monthly_t2m.nc')
clipData(cesm_precip_data,False).to_netcdf('cesm_boreal_monthly_tp.nc')
