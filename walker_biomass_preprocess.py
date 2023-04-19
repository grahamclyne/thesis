#preprocess agb walker data
import rioxarray 
import xarray
import geopandas as gpd
from preprocessing.utils import scaleLongitudes
from hydra import initialize, compose

with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config")
# nfis_tif = rioxarray.open_rasterio(f'{self.cfg.data}/CA_forest_total_biomass_2015_NN/CA_forest_total_biomass_2015.tif',lock=False)

soil_df = rioxarray.open_rasterio('/Users/gclyne/thesis/data/Base_Cur_AGB_MgCha_500m.tif')
ref_df = xarray.open_dataset('/Users/gclyne/thesis/data/cesm/cSoilAbove1m_Emon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc')
ref_df = ref_df.rio.set_crs('epsg:4326')
soil_df = soil_df.rio.reproject_match(ref_df)
soil_df = soil_df.rename({'x':'lon','y':'lat'})
canada_mf_shapefile = gpd.read_file("data/shapefiles/ecozones.shp")
canada_mf_shapefile.to_crs(soil_df.rio.crs, inplace=True)
scaled_soil = scaleLongitudes(soil_df)
scaled_soil = scaled_soil.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
x = scaled_soil.rio.clip(canada_mf_shapefile.geometry.apply(lambda x: x.__geo_interface__), canada_mf_shapefile.crs, drop=True, invert=False, all_touched=False, from_disk=False)
ds_masked = x.where(x.data != x.rio.nodata)  
soil_pdf = ds_masked.sel(band=1).to_pandas()


soil_pdf = ds_masked.sel(band=1).to_dataframe(name='soil')
soil_pdf = soil_pdf.rename(columns={'soil':'cSoilAbove1m'})
soil_pdf.reset_index(inplace=True)
soil_pdf.drop(columns=['band','spatial_ref'],inplace=True)
soil_pdf.dropna(inplace=True)
from preprocessing.utils import getArea
soil_pdf['area'] = soil_pdf.apply(lambda x: getArea(x['lat'],x['lon'],cfg),axis=1)
(soil_pdf['area'] / 10000 * soil_pdf['cSoilAbove1m']).sum() / 1e6