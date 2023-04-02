import xarray as xr
import rioxarray

regrowth = rioxarray.open_rasterio(f'/Users/gclyne/Downloads/CA_forest_harvest_years2recovery/CA_forest_harvest_years2recovery.tif',decode_coords='all',lock=False,chunks=True)
regrowth.rio.reproject('EPSG:4326').rio.to_raster(f'/Users/gclyne/Downloads/CA_forest_harvest_years2recovery/CA_forest_harvest_years2recovery_reprojected.tif')
