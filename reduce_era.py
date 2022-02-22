import xarray as xr
import cfgrib

# need to export ECCODES_DIR='/home/graham/eccodes'
# era = xr.load_dataset('/home/graham/Downloads/era_temp_precip_1984_2015.grib',filter_by_keys={'stepType': 'avgid'})
# x = era.groupby('time.year').mean('time')

# x.to_netcdf('era_groupedby_year.nc')
tp_era_1984_2001 = xr.open_dataset('/home/graham/Downloads/era_temp_precip_1984_2015.grib',filter_by_keys={'stepType': 'avgas'})
tp_era_2002_2015 = xr.open_dataset('/home/graham/Downloads/era_temp_precip_1984_2015.grib',filter_by_keys={'stepType': 'avgas'})
t2m_era_1984_2015 = xr.open_dataset('/home/graham/Downloads/era_temp_precip_1984_2015.grib',filter_by_keys={'stepType': 'avgid'})
print(tp_era_1984_2001['tp'])
print(type(tp_era_2002_2015))
tp = xr.merge([tp_era_2002_2015,tp_era_1984_2001])
tp = tp.groupby('time.year').mean('time')
tp.to_netcdf('tp_groupedby_year.nc')
t2m = t2m_era_1984_2015.groupby('time.year').mean('time')
t2m.to_netcdf('t2m_groupedby_year.nc')
