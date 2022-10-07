import pandas as pd
import numpy as np
from utils import borealCoordinates
import xarray as xr
import config
import math

#open relevation files
FORC_PATH = '/Users/gclyne/thesis/data/FORC'
vars = pd.read_csv(f'{FORC_PATH}/ForC_variables.csv')
sites = pd.read_csv(f'{FORC_PATH}/ForC_sites.csv',encoding='ISO-8859-1')
plots = pd.read_csv(f'{FORC_PATH}/ForC_plots.csv')
pft = pd.read_csv(f'{FORC_PATH}/ForC_pft.csv')
methodology = pd.read_csv(f'{FORC_PATH}/ForC_methodology.csv')
histtype = pd.read_csv(f'{FORC_PATH}/ForC_histtype.csv')
history = pd.read_csv(f'{FORC_PATH}/ForC_history.csv',encoding='ISO-8859-1')
allometry = pd.read_csv(f'{FORC_PATH}/ForC_allometry.csv')
citations = pd.read_csv(f'{FORC_PATH}/ForC_citations.csv',encoding='ISO-8859-1')
measurements = pd.read_csv(f'{FORC_PATH}/ForC_measurements.csv')

veg = xr.open_dataset(f'{config.CESM_PATH}/cVeg_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc').groupby('time.year').mean()
cwd = xr.open_dataset(f'{config.CESM_PATH}/cCwd_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc').groupby('time.year').mean()
litter = xr.open_dataset(f'{config.CESM_PATH}/cLitter_Lmon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc').groupby('time.year').mean()
soil = xr.open_dataset(f'{config.CESM_PATH}/cSoil_Emon_CESM2_land-hist_r1i1p1f1_gn_185001-201512.nc').groupby('time.year').mean()


def getGridCellValue(netcdf:xr.Dataset,variable:str,lat:float,lon:float,year:int) -> float:
    val = netcdf.sel(lat=lat,lon=lon,method='nearest',year=year)[variable].item()
    print(val,lat,lon,year,variable)
    return val
def findCoordinateMatch(row,boreal_coordinates):
    for lat,lon in boreal_coordinates:
        if row[1] >= lat and row[2] >= lon and row[1] < (lat + 1) and row[2] < (lon + 1.25):
            return [row[0],lat,lon]
    return [row[0],np.nan,np.nan]


boreal_coordinates = borealCoordinates()
#get relevant columns from measurements
measure_subset = measurements[['sites.sitename','variable.name','date','start.date','end.date','stand.age','dominant.veg','mean','area.sampled']]

#get coordinates for each measurement 
x = measure_subset.merge(sites[['sites.sitename','lat','lon']], how='left', left_on='sites.sitename', right_on='sites.sitename')

matches = []

site_subset = sites[['sites.sitename','lat','lon']]
#get grid cell coordinates from boreal_coordinates
numpy_sites = site_subset.to_numpy()

#get grid cell matches
grid_lat_lon_coordinates = []
for row in numpy_sites: 
    matches.append(findCoordinateMatch(row,boreal_coordinates))

coord = pd.DataFrame(matches)
coord.columns = ('sites.sitename','grid_lat','grid_lon')
coords = coord.merge(sites[['sites.sitename','lat','lon']],how='left',left_on='sites.sitename', right_on='sites.sitename')

x = measure_subset.merge(coords, how='left', left_on='sites.sitename', right_on='sites.sitename')
reduced = x.dropna(subset=['grid_lat', 'grid_lon'])
reduced_relevant = reduced.merge(vars[['variable.name','variable.type','units']], how='left',left_on='variable.name',right_on='variable.name')
reduced_relevant = reduced_relevant[reduced_relevant['variable.type'] == 'stock']


modeled_carbon_stocks = []
reduced_relevant = reduced_relevant.reset_index().drop(['index'],axis=1)
coordinates_for_iter = reduced_relevant[['grid_lat','grid_lon','date']].to_numpy()

#find carbon stock values for each coordinate
for lat,lon,year in coordinates_for_iter:
    if(year == 'NRA' or year == 'NI' or year == 'NAC' or math.isnan(float(year))):
        year = 2000
    else:
        year = round(float(year))
    modeled_carbon_stocks.append([
        getGridCellValue(veg,'cVeg',lat,lon%360,year),
        getGridCellValue(cwd,'cCwd',lat,lon%360,year),
        getGridCellValue(litter,'cLitter',lat,lon%360,year),
        getGridCellValue(soil,'cSoil',lat,lon%360,year)
    ])


modeled_carbon_df = pd.DataFrame(modeled_carbon_stocks)
modeled_carbon_df.columns = ('vegetation','cwd','litter','soil')

#assign columns to main
reduced_relevant['soil'] = modeled_carbon_df['soil']
reduced_relevant['vegetation'] = modeled_carbon_df['vegetation']
reduced_relevant['cwd'] = modeled_carbon_df['cwd']
reduced_relevant['litter'] = modeled_carbon_df['litter']

reduced_relevant.to_csv('forc_cesm_validation.csv')