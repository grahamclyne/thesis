

from osgeo import gdal
import pandas as pd
import numpy as np 
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def load_data(FILEPATH):
    ds = gdal.Open(FILEPATH)
    print(ds)
    return ds

# Opens the data HDF file and returns as a dataframe
def read_dataset(SUBDATASET_NAME, FILEPATH):
    dataset = load_data(FILEPATH)
    path = ''
    for i in dataset.GetSubDatasets():
        print(i)
    for sub, description in dataset.GetSubDatasets():
        if (description.endswith(SUBDATASET_NAME)):
            path = sub
            break
    if(path == ''):
        print(SUBDATASET_NAME + ' not found')
        return
    subdataset = gdal.Open(path)
    print(subdataset.GetMetadata())
    subdataset = subdataset.ReadAsArray()
    #subdataset = pd.DataFrame(subdataset)
    return subdataset


ds = read_dataset("Day_view_angl MODIS_Grid_8Day_1km_LST (8-bit unsigned integer)","MOD11A2.A2021321.h12v02.006.2021331131605.hdf")
test = np.array(ds[:,:], np.float)
m = Basemap(projection='sinu', resolution = 'c',
    lon_0=-110)
cdict = {'red' : [(0,0.,0.), (100./255.,1.,1.),(1.,0.,0.)],
         'green' : [(0,0.,0.),(1.,0.,0.)] , 
         'blue' : [(0,0.,0.),(100./255.,0.,0.),(1.,1.,1.)] }
blue_red = LinearSegmentedColormap('BlueRed',cdict)

#m.drawcoastlines(linewidth=0.5)
#m.drawparallels(np.arange(-90,120,30), labels=[1,0,0,0])
#m.drawmeridians(np.arange(-180,181,45),labels=[0,0,0,1])
m.imshow(np.flipud(ds))
plt.show()
# replace <subdataset> with the number of the subdataset you need, starting with 0


# from pyhdf import SD
# import numpy as np
# from matplotlib import pyplot as plt
# #from mpl_toolkits.basemap import Basemap
# from matplotlib.colors import LinearSegmentedColormap
# hdf = SD.SD('MOD11A2.A2021321.h12v02.006.2021331131605.hdf')
# data = hdf.select('Eight_Day_CMG_Snow_Cover')
# snowcover=np.array(data[:,:],np.float)
# snowcover[np.where(snowcover==255)]=np.nan
# m = Basemap(projection='cyl', resolution = 'l',
#     llcrnrlat=-90, urcrnrlat=90,llcrnrlon=-180,urcrnrlon=180)
# cdict = {'red' : [(0,0.,0.), (100./255.,1.,1.),(1.,0.,0.)],
#          'green' : [(0,0.,0.),(1.,0.,0.)] , 
#          'blue' : [(0,0.,0.),(100./255.,0.,0.),(1.,1.,1.)] }
# blue_red = LinearSegmentedColormap('BlueRed',cdict)

# m.drawcoastlines(linewidth=0.5)
# m.drawparallels(np.arange(-90,120,30), labels=[1,0,0,0])
# m.drawmeridians(np.arange(-180,181,45),labels=[0,0,0,1])
# m.imshow(np.flipud(snowcover),cmap=blue_red)
# plt.title('MOD10C2: Eight Day Global Snow Cover')
# plt.show()