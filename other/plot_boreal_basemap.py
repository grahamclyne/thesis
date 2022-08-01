from osgeo import gdal, gdalnumeric, ogr, osr
import numpy as np
import csv
import xarray as xr
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import matplotlib.pyplot as plt
import geopandas as gpd 
import rioxarray as rio
from shapely.geometry import mapping

# print(len(np.unique(lats)))
# unique_lats = np.unique(lats)
# # print(len(np.unique(lons)))
# unique_lons = np.unique(lons)
# topoin.reshape((44,112))
data= np.empty((192,288))
data[:] = np.NaN
tree_cover_data = xr.open_dataset('/home/graham/code/thesis/data/treeFrac_Lmon_CESM2_land-hist_r1i1p1f1_gn_194901-201512.nc')
lats = tree_cover_data['lat'].data
lons = tree_cover_data['lon'].data
data_row = []
with open('/home/graham/boreal_observed_output.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        data_row.append((float(row[0]),float(row[1]),float(row[2])))
for l,t,d in data_row:
    lat_index = np.abs(lats - l).argmin()
    # print(l,t,d)
    lon_index = np.abs(lons - (t + 360)).argmin()
    if(d != 0):
        data[lat_index,lon_index] = d
    
topoin,lons = shiftgrid(180.,data,lons,start=False)



fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
# setup of basemap ('lcc' = lambert conformal conic).
# use major and minor sphere radii from WGS84 ellipsoid.
m = Basemap(llcrnrlon=-145.5,llcrnrlat=38,urcrnrlon=-35,urcrnrlat=60,\
            rsphere=(6378137.00,6346752.3142),\
            resolution='f',area_thresh=1000.,projection='lcc',\
            lat_1=60.,lon_0=-107.,ax=ax)
m.readshapefile(r'/home/graham/code/thesis/data/boreal_reduced', 'states', drawbounds = True,color='red')

# transform to nx x ny regularly spaced 100km native projection grid
nx = int((m.xmax-m.xmin)/100000.)+1; ny = int((m.ymax-m.ymin)/100000.)+1 
topodat = m.transform_scalar(topoin,lons,lats,nx,ny)
print(type(topodat))
# m.bluemarble()

# plot image over map with imshow.
im = m.imshow(topodat)
# draw coastlines and political boundaries.
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
ax.set_title('Observed Tree Cover in North American Boreal Forest (1984)')
plt.legend()
plt.savefig('output_observed')
