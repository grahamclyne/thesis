
import cftime
import xarray as xr
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import numpy as np


fn = '/home/graham/Downloads/treeFrac_Lmon_CESM2_land-hist_r1i1p1f1_gn_194901-201512.nc'

dset = xr.open_dataset(fn)
t = cftime.DatetimeNoLeap(2001, 1, 1)
#need to use longitude 0-360, convert from -180 to 180
ffrac = dset['treeFrac'].sel(lat=55,lon=235,time=t, method='nearest')
# print(ffrac)
fp = r'Hansen_GFC-2020-v1.8_lossyear_60N_130W.tif'
tree_cover = r'Hansen_GFC-2020-v1.8_treecover2000_60N_130W.tif'
img = rasterio.open(fp)
tree_cover_img = rasterio.open(tree_cover)
band1 = img.read(1)
band2 = tree_cover_img.read(1)
#print(img.bounds.left, img.bounds.bottom, img.bounds.top,img.bounds.left)
x,y = img.index(-125,55.13089)
x1,y1 = img.index(-126.25,56.073299)
# print(x, y)
# print(x1, y1)
# print(img.xy(0,0))
# print(img.xy(x,y))
cropped = band1[x1:x,y1:y]
tree_cover_cropped = band2[x1:x, y1:y]
print(tree_cover_cropped)
cropped[cropped != 1] = 0
# print(cropped)
# print(cropped.shape)
# print(tree_cover_cropped.shape)
tree_loss_2001 = tree_cover_cropped[cropped>0]
# print(tree_loss_2001.shape)
percentage_2001 = tree_loss_2001.sum() / (tree_loss_2001.shape[0])
percentage_2000 = tree_cover_cropped.sum() / (tree_cover_cropped.shape[0] * tree_cover_cropped.shape[1])
# print(percentage_2001, percentage_2000)
#print(count / (band1.shape[0] * band1.shape[1]))

# for i in band1[:,-1]:
#     print(i)


# lat bounds  for test 55.13089 ,  56.073299
# long bounds (-125, -126.25)