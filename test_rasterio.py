import rasterio as rio
import rioxarray
obs_tree = rioxarray.open_rasterio('/Users/gclyne/thesis/recast_1984.nc')
print(obs_tree.sel(band=1))
print(obs_tree.data)
