from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import cftime
import rasterio 
import geopandas
from sklearn import tree
# read in etopo5 topography/bathymetry.
# url = 'http://ferret.pmel.noaa.gov/thredds/dodsC/data/PMEL/etopo5.nc'
# etopodata = Dataset(url)
import xarray as xr
tree_cover_data = xr.open_dataset('/home/graham/code/thesis/data/treeFrac_Lmon_CESM2_land-hist_r1i1p1f1_gn_194901-201512.nc')
# latitude = tree_cover_data['lat'].data
# longitude = tree_cover_data['lon'].data
# mean_data = tree_cover_data['treeFrac'].isel(time=801).data
# mean_data_array = tree_cover_data['treeFrac'].isel(time=801)
# lat_res = latitude[10] - latitude[9]

topoin = tree_cover_data['treeFrac'].sel(time=cftime.DatetimeNoLeap(1984, 1, 1),method='nearest').data

lons = tree_cover_data['lon'].data
lats = tree_cover_data['lat'].data
# shift data so lons go from -180 to 180 instead of 20 to 380.
# sf = geopandas.read_file('/home/graham/Downloads/gpr_000b11a_e.shp')

# ShapeMask = rasterio.features.geometry_mask(sf.iloc[0],
#                                       out_shape=(len(lats), len(lons)),
#                                       transform=tree_cover_data.geobox.transform,
#                                       invert=True)
# ShapeMask = xr.DataArray(ShapeMask , dims=("lat", "lon"))

# # Then apply the mask
# topoin = tree_cover_data.where(ShapeMask == True).data

topoin,lons = shiftgrid(180.,topoin,lons,start=False)

# plot topography/bathymetry as an image.
print(topoin)
print(lons)
print(lats)
# create the figure and axes instances.
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
# setup of basemap ('lcc' = lambert conformal conic).
# use major and minor sphere radii from WGS84 ellipsoid.
m = Basemap(llcrnrlon=-145.5,llcrnrlat=38,urcrnrlon=-35,urcrnrlat=60,\
            rsphere=(6378137.00,6346752.3142),\
            resolution='f',area_thresh=1000.,projection='lcc',\
            lat_1=60.,lon_0=-107.,ax=ax)
m.readshapefile(r'/home/graham/code/thesis/data/boreal_reduced', 'states', drawbounds = True)

# transform to nx x ny regularly spaced  native projection grid
nx = int((m.xmax-m.xmin)/100000.)+1; ny = int((m.ymax-m.ymin)/100000.)+1
topodat = m.transform_scalar(topoin,lons,lats,nx,ny)
m.bluemarble()
# plot image over map with imshow.
im = m.imshow(topodat,alpha=)
# draw coastlines and political boundaries.
# m.drawstates()
# draw parallels and meridians.
# label on left and bottom of map.
parallels = np.arange(0.,80,20.)
# m.drawparallels(parallels)
meridians = np.arange(10.,360.,30.)
# m.drawmeridians(meridians,labels=[1,0,0,1])
m.drawmapboundary(fill_color='aqua')

# add colorbar
cb = m.colorbar(im,"right", size="5%", pad='2%')
cb.ax.set_ylabel("% of tree cover")
ax.set_title('Simulated Tree Cover in North American Boreal Forest (1984)')
plt.savefig("output")
