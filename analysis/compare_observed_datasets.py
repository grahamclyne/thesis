import pandas as pd
import hydra 
from omegaconf import DictConfig
from preprocessing.utils import getArea,getGeometryBoxes,scaleLongitudes
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import scipy.stats.pearsonr
import numpy as np
import rioxarray
import xarray 
import matplotlib.colors as colors
from lstm_model import lat_adjusted_mse
import scipy


def preprocessSoil(cfg):
    soil_df = rioxarray.open_rasterio('/Users/gclyne/thesis/data/McMaster_WWFCanada_soil_carbon1m_250m_kg-m2_version1.0.tif')
    ref_df = xarray.open_dataset('/Users/gclyne/thesis/data/cesm/cSoilAbove1m_Emon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc')
    ref_df = ref_df.rio.set_crs('epsg:4326')
    soil_df = soil_df.rio.reproject_match(ref_df)
    soil_df = soil_df.rename({'x':'lon','y':'lat'})
    ecozones_shapefile = gpd.read_file("data/shapefiles/Ecozones.shp")
    ecozones_shapefile.to_crs(soil_df.rio.crs, inplace=True)
    scaled_soil = scaleLongitudes(soil_df)
    scaled_soil = scaled_soil.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=False)
    x = scaled_soil.rio.clip(ecozones_shapefile.geometry.apply(lambda x: x.__geo_interface__), ecozones_shapefile.crs, drop=True, invert=False, all_touched=False, from_disk=False)
    ds_masked = x.where(x.data != x.rio.nodata)  
    soil_pdf = ds_masked.sel(band=1).to_pandas()
    soil_pdf = ds_masked.sel(band=1).to_dataframe(name='soil')
    soil_pdf = soil_pdf.rename(columns={'soil':'cSoilAbove1m'})
    soil_pdf.reset_index(inplace=True)
    soil_pdf.drop(columns=['band','spatial_ref'],inplace=True)
    soil_pdf.dropna(inplace=True)
    return soil_pdf 

def getRegionalValues(df:pd.DataFrame,ecozones_coords:pd.DataFrame,cfg,variable:str) -> pd.DataFrame:
    # df['lat'] = df['lat'].round(6)
    # # print(df['lat'].unique())
    # print(ecozones_coords['lat'].unique())
    df = pd.merge(df,ecozones_coords,on=['lat','lon'],how='inner')
    # print(df)
    # if(variable != 'treeFrac'):
    #     df['area'] = df.apply(lambda x: getArea(x['lat'],x['lon'],cfg),axis=1)
    #     df[variable] = df[variable] * df['area'] /1e9 # make non-spatial,covnert to megatonnes
    #unit is now mtc 
    return df

def pandasToGeo(df:pd.DataFrame,variable:str) -> gpd.GeoDataFrame:
    boxes = getGeometryBoxes(df)
    df = gpd.GeoDataFrame(df[[variable,'year']], geometry=boxes)
    return df

#AGB FIGURE FOR ESTIMATE COMPARISON
def plotComparison(dataframes:list,canada:gpd.GeoDataFrame,ecozones:gpd.GeoDataFrame,titles:list,filename:str,column_name:str) -> None:
    if (len(dataframes) == 4):
        f, axes = plt.subplots(figsize=(30, 20),nrows=int(len(dataframes)/2),ncols=int(len(dataframes)/2))
        # plt.subplots_adjust(left=0.0,
        #                     bottom=0.5,
        #                     right=0.1,
        #                     top=1.1,
        #                     wspace=0.1,
        #                     hspace=0)
    else:
        f, axes = plt.subplots(figsize=(30, 8),nrows=1,ncols=len(dataframes))
    max_val = max([x[column_name].max() for x in dataframes])
    min_val = min([x[column_name].min() for x in dataframes])
    min_val = 0
    max_val = 1000
    axes = axes.flatten()
    norm = mpl.colors.SymLogNorm(1,vmin=min_val,vmax=max_val)
    for ax_index in range(0,len(axes)):
        canada.plot(ax=axes[ax_index],alpha=0.1)
        ecozones.plot(ax=axes[ax_index],color='white',edgecolor='black',alpha=0.1)
        ax = dataframes[ax_index].plot(ax=axes[ax_index],column=column_name,norm=norm,cmap='Greens')
        ax.set_xlabel('Longitude',fontsize=35)
        ax.set_ylabel('Latitude',fontsize=35)
        ax.tick_params(axis='both', which='major', labelsize=40)
        
        x = mpl.image.AxesImage(ax=axes[ax_index])
        axes[ax_index].title.set_text(titles[ax_index])
        axes[ax_index].title.set_fontsize(35)
    m = plt.cm.ScalarMappable(cmap='Greens',norm=norm)
    # m.set_array(norm(dataframes[0][column_name]))
    cbar = plt.colorbar(m,fraction=0.026, pad=0.04)
    cbar.ax.set_ylabel(f'{column_name.capitalize()} (kg/m2)',fontsize=35)
    cbar.ax.tick_params(labelsize=30)
    f.tight_layout()
    plt.savefig(f'figures/{filename}.png')

def plotSoilComparison(dataframes:list,canada:gpd.GeoDataFrame,ecozones:gpd.GeoDataFrame,titles:list,filename:str,column_name:str,units:str) -> None:

    f, axes = plt.subplots(figsize=(30, 8),nrows=1,ncols=len(dataframes))

    axes = axes.flatten()
    max_val = np.argmax([x[column_name].max() for x in dataframes])
    min_val = np.argmin([x[column_name].min() for x in dataframes])
    min_val = 0
    max_val = 1000
    # print(dataframes[max_val])
    # print(dataframes)
    # print(min(dataframes[max_val].min()))
    # print(max(dataframes[max_val]))
    # norm = mpl.colors.SymLogNorm(1,vmin=dataframes[max_val][column_name].min(),vmax=dataframes[max_val][column_name].max())
    norm = mpl.colors.SymLogNorm(1,vmin=min_val,vmax=max_val)
    for ax_index in range(0,len(axes)):   
        # norm = mpl.colors.Normalize(dataframes[ax_index].min(),dataframes[ax_index].max(),clip=True)
        canada.plot(ax=axes[ax_index],alpha=0.1)
        ecozones.plot(ax=axes[ax_index],color='white',edgecolor='black',alpha=0.1)
        ax = dataframes[ax_index].plot(ax=axes[ax_index],column=column_name,norm=norm,cmap='Greens')
        ax.set_xlabel('Longitude',fontsize=30)
        ax.set_ylabel('Latitude',fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=30)
        
        x = mpl.image.AxesImage(ax=axes[ax_index])
        axes[ax_index].title.set_text(titles[ax_index])
        axes[ax_index].title.set_fontsize(30)
    m = plt.cm.ScalarMappable(cmap='Greens',norm=norm)
    # m.set_array(dataframes[0][column_name])
    cbar = plt.colorbar(m,fraction=0.026, pad=0.04,ax=ax,norm=norm)
    cbar.ax.set_ylabel(f'{column_name} ({units})',fontsize=30)
    cbar.ax.tick_params(labelsize=30)

    f.tight_layout()
    # f.suptitle(f'Comparison of Soil Carbon Estimates',fontsize=60)
    plt.savefig(f'figures/{filename}.png')

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):

    list_of_regions = ['Boreal Shield','Boreal Cordillera','Boreal PLain']

    #prepare data
    soil_pdf = preprocessSoil(cfg)
    nfis_agb = pd.read_csv(f'{cfg.data}/nfis_agb_df.csv')
    walker_agb = pd.read_csv(f'{cfg.data}/walker_agb_df.csv')
    emulated = pd.read_csv(f'{cfg.data}/emulation_df.csv')
    emulated = emulated[emulated['year'] == 2015]
    cesm_data = pd.read_csv(f'{cfg.data}/cesm_data_variant.csv')
    cesm_data = cesm_data.groupby(['year','lat','lon']).mean().reset_index()     #transform from rolling window to yearly average
    cesm_data = cesm_data[cesm_data['year'] == 2014]
    input_data = pd.read_csv(f'{cfg.data}/observed_timeseries{cfg.model.seq_len}_data.csv')
    input_data = input_data.groupby(['year','lat','lon']).mean().reset_index()  
    input_data = input_data[input_data['year'] == 2014]
    print(input_data.head())
    ecozones_coords = pd.read_csv(f'{cfg.data}/ecozones_coordinates.csv')
    ecozones_coords = ecozones_coords[ecozones_coords['zone'].isin(list_of_regions)]

    nfis_agb['agb'] = nfis_agb['agb'] / 10 #initally in tonnes/hectare, need to convert to kg/m2 
    walker_agb['agb'] = walker_agb['agb'] / 10 #initally in Megagrams/hectare, need to convert to kg/m2
    emulated['agb'] = emulated['cStem'] + emulated['cOther'] + emulated['cLeaf'] #this is in kg/m2
    cesm_data['agb'] = cesm_data['cStem'] + cesm_data['cOther'] + cesm_data['cLeaf'] #this is in kg/m2

    walker_agb['lat'] = walker_agb['lat'].round(6)
    nfis_agb['lat'] = nfis_agb['lat'].round(6)
    emulated['lat'] = emulated['lat'].round(6)
    cesm_data['lat'] = cesm_data['lat'].round(6)
    ecozones_coords['lat'] = ecozones_coords['lat'].round(6)
    soil_pdf['lat'] = soil_pdf['lat'].round(6)
    input_data['lat'] = input_data['lat'].round(6)

    #emulated has smallest area due to ERA nodata, so we need to merge on that
    walker_agb = pd.merge(walker_agb,emulated,on=['lat','lon'],how='inner',suffixes=('','_emulated'))
    nfis_agb = pd.merge(nfis_agb,emulated,on=['lat','lon'],how='inner',suffixes=('','_emulated'))
    cesm_data = pd.merge(cesm_data,emulated,on=['lat','lon'],how='inner',suffixes=('','_emulated'))
    soil_pdf = pd.merge(soil_pdf,emulated,on=['lat','lon'],how='inner',suffixes=('','_emulated'))
    input_data = pd.merge(input_data,emulated,on=['lat','lon'],how='inner',suffixes=('','_emulated'))

    #soil has missing data, so merge to remove areas that are missing
    cesm_data_soil = pd.merge(cesm_data,soil_pdf,on=['lat','lon'],how='inner',suffixes=('','_soil'))
    emulated_soil = pd.merge(emulated,soil_pdf,on=['lat','lon'],how='inner',suffixes=('','_soil'))

    #get regional AGB
    regional_nfis_agb = getRegionalValues(nfis_agb,ecozones_coords,cfg,'agb')
    regional_walker_agb = getRegionalValues(walker_agb,ecozones_coords,cfg,'agb')
    regional_emulated_agb = getRegionalValues(emulated,ecozones_coords,cfg,'agb')
    regional_cesm_agb = getRegionalValues(cesm_data,ecozones_coords,cfg,'agb')
    regional_sothe_soil = getRegionalValues(soil_pdf,ecozones_coords,cfg,'cSoilAbove1m')
    regional_cesm_soil = getRegionalValues(cesm_data_soil,ecozones_coords,cfg,'cSoilAbove1m')
    regional_emulated_soil = getRegionalValues(emulated_soil,ecozones_coords,cfg,'cSoilAbove1m')
    #shape files for plotting
    canada = gpd.read_file(f'{cfg.data}/shapefiles/lpr_000b16a_e/lpr_000b16a_e.shp')
    canada = canada.to_crs('4326')
    ecozones = gpd.read_file('data/shapefiles/ecozones.shp').to_crs('epsg:4326')
    ecozones = ecozones.where(ecozones['ZONE_NAME'].isin(['Boreal Shield','Boreal Cordillera','Boreal PLain']))

    #convert to geodataframes
    nfis_gdf = pandasToGeo(regional_nfis_agb,'agb')
    walker_gdf = pandasToGeo(regional_walker_agb,'agb')
    emulated_gdf = pandasToGeo(regional_emulated_agb,'agb')
    cesm_gdf = pandasToGeo(regional_cesm_agb,'agb')
    soil_gdf = pandasToGeo(regional_sothe_soil,'cSoilAbove1m')
    cesm_soil_gdf = pandasToGeo(regional_cesm_soil,'cSoilAbove1m')
    emulated_soil_gdf = pandasToGeo(regional_emulated_soil,'cSoilAbove1m')


    # for var in cfg.model.input:
    #     regional_observed = pd.merge(input_data,ecozones_coords,on=['lat','lon'],how='inner')
    #     regional_cesm = pd.merge(cesm_data,ecozones_coords,on=['lat','lon'],how='inner')

    #     regional_observed_gdf = pandasToGeo(regional_observed,var)
    #     regional_cesm_gdf = pandasToGeo(regional_cesm,var)
    #     difference = regional_observed_gdf.copy()
    #     difference[var] = regional_observed_gdf[var] - regional_cesm_gdf[var]
    #     units = {'tas_JJA':'K','tas_DJF':'K','pr':'kg m-2 s-1','ps':'Pa','treeFrac':'%','tsl':'K'}
    #     plotSoilComparison([regional_observed_gdf,regional_cesm_gdf,difference],canada,ecozones,['ERANFIS (2014)','CESM2 (2014)','Difference'],f'observed_{var}_cesm_comparison',var,units[var])
    
    plotComparison([nfis_gdf,walker_gdf,emulated_gdf,cesm_gdf],canada,ecozones,['Matasci et al. (year=2015)','Walker et al. (year=2015)','ERANFIS/Emulation (year=2015)','CESM2 Model Output (year=2014)'],'agb_comparison','agb')
    plotSoilComparison([soil_gdf,emulated_soil_gdf,cesm_soil_gdf],canada,ecozones,['Sothe et al. (year=2015)','ERANFIS/Emulation (year=2015)','CESM2 Model Output (year=2014)'],'soil_comparison','cSoilAbove1m','kg/m2')
    plotComparison([cesm_gdf,emulated_gdf],canada,ecozones,['CESM2 Model Output (year=2014)','ERANFIS/Emulation (year=2015)'],'cesm_emulated_comparison','agb')
    plotSoilComparison([cesm_soil_gdf,emulated_soil_gdf],canada,ecozones,['CESM2 Model Output (year=2014)','ERANFIS/Emulation (year=2015)'],'cesm_emulated_soil_comparison','cSoilAbove1m','kg/m2')
    #REGIONAL SUMS
    for region in list_of_regions:
        print(f'For {region}')
        print(
            f'{round(regional_nfis_agb[regional_nfis_agb["zone"]==region]["agb"].sum(),2)} & \
        {round(regional_walker_agb[regional_walker_agb["zone"]==region]["agb"].sum(),2)} & \
        {round(regional_emulated_agb[regional_emulated_agb["zone"]==region]["agb"].sum(),2)} & \
        {round(regional_cesm_agb[regional_cesm_agb["zone"]==region]["agb"].sum(),2)} & \
        {round(regional_sothe_soil[regional_sothe_soil["zone"]==region]["cSoilAbove1m"].sum(),2)} & \
        {round(regional_emulated_soil[regional_emulated_soil["zone"]==region]["cSoilAbove1m"].sum(),2)} & \
        {round(regional_cesm_soil[regional_cesm_soil["zone"]==region]["cSoilAbove1m"].sum(),2)} & ')

    #TOTAL SUMS
    nfis_sum = round(regional_nfis_agb['agb'].sum(),2)
    walker_sum = round(regional_walker_agb['agb'].sum(),2)
    emulated_sum = round(regional_emulated_agb['agb'].sum(),2)
    cesm_sum = round(regional_cesm_agb['agb'].sum(),2)
    cesm_soil_sum = round(regional_cesm_soil['cSoilAbove1m'].sum(),2)
    emulated_soil_sum = round(regional_emulated_soil['cSoilAbove1m'].sum(),2)
    soil_sum = round(regional_sothe_soil['cSoilAbove1m'].sum(),2)

    print(f'{nfis_sum} & {walker_sum} & {emulated_sum} & {cesm_sum} & {soil_sum} & {emulated_soil_sum} & {cesm_soil_sum} \\\\')

    # #merge pred and sothe and visualize r2 value
    # def visualize_r2(predicted,actual,name):
    #     fig, ax = plt.subplots(figsize=(10, 10))
    #     ax.scatter(predicted, actual, s=100, edgecolor="black",linewidth=0.5)
    #     ax.set_xlabel("Predicted")
    #     ax.set_ylabel("Observed")
    #     ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'k--', lw=4)
    #     ax.plot([predicted.min(), predicted.max()], [predicted.min(), predicted.max()], 'k--', lw=4)

    #     ax.annotate("r-squared = {:.3f}".format(scipy.stats.pearsonr(actual, predicted)), (0, 1))

    #     plt.savefig(f'figures/{name}_r2.png')

    #R2 value of emulation,CESM compared to NFIS, walker 
    print(f'R2 value of emulation compared to NFIS: {round(scipy.stats.pearsonr(regional_nfis_agb["agb"],regional_emulated_agb["agb"]).statistic,2)}')
    print(f'R2 value of emulation compared to Walker: {round(scipy.stats.pearsonr(regional_walker_agb["agb"],regional_emulated_agb["agb"]).statistic,2)}')
    print(f'R2 value of CESM compared to NFIS: {round(scipy.stats.pearsonr(regional_nfis_agb["agb"],regional_cesm_agb["agb"]).statistic,2)}')
    print(f'R2 value of CESM compared to Walker: {round(scipy.stats.pearsonr(regional_walker_agb["agb"],regional_cesm_agb["agb"]).statistic, 2)}')
    print(f'R2 value of CESM compared to Sothe: {round(scipy.stats.pearsonr(regional_sothe_soil["cSoilAbove1m"],regional_cesm_soil["cSoilAbove1m"]).statistic, 2)}')
    print(f'R2 value of Emulated compared to Sothe: {round(scipy.stats.pearsonr(regional_sothe_soil["cSoilAbove1m"],regional_emulated_soil["cSoilAbove1m"]).statistic, 2)}')
    print(f'R2 value of Emulated compared to CESM2: {round(scipy.stats.pearsonr(regional_cesm_agb["agb"],regional_emulated_agb["agb"]).statistic,2)}')
    print(f'R2 value of Emulated compared to CESM2 SOIL: {round(scipy.stats.pearsonr(regional_cesm_soil["cSoilAbove1m"],regional_emulated_soil["cSoilAbove1m"]).statistic,2)}')

    # visualize_r2(regional_emulated_agb["agb"],regional_nfis_agb["agb"],'emulated_nfis')
    # visualize_r2(regional_emulated_agb["agb"],regional_walker_agb["agb"],'emulated_walker')
    # visualize_r2(regional_cesm_agb["agb"],regional_nfis_agb["agb"],'cesm_nfis')
    # visualize_r2(regional_cesm_agb["agb"],regional_walker_agb["agb"],'cesm_walker')
    # visualize_r2(regional_cesm_soil["cSoilAbove1m"],regional_sothe_soil["cSoilAbove1m"],'cesm_sothe')
    # visualize_r2(regional_emulated_soil["cSoilAbove1m"],regional_sothe_soil["cSoilAbove1m"],'emulated_sothe')
    # visualize_r2(regional_emulated_agb["agb"],regional_cesm_agb["agb"],'emulated_cesm')
    import torch
    #RMSE values of emulation,CESM compared to NFIS, walker
    print(f'MSE value of emulation compared to NFIS: {torch.round(lat_adjusted_mse(torch.tensor(regional_nfis_agb["agb"]),torch.tensor(regional_emulated_agb["agb"]),torch.tensor(regional_emulated_agb["lat"])),decimals=2)}')
    print(f'MSE value of emulation compared to Walker: {torch.round(lat_adjusted_mse(torch.tensor(regional_walker_agb["agb"]),torch.tensor(regional_emulated_agb["agb"]),torch.tensor(regional_emulated_agb["lat"])),decimals=2)}')
    print(f'MSE value of CESM compared to NFIS: {torch.round(lat_adjusted_mse(torch.tensor(regional_nfis_agb["agb"]),torch.tensor(regional_cesm_agb["agb"]),torch.tensor(regional_emulated_agb["lat"])),decimals=2)}')
    print(f'MSE value of CESM compared to Walker: {torch.round(lat_adjusted_mse(torch.tensor(regional_walker_agb["agb"]),torch.tensor(regional_cesm_agb["agb"]),torch.tensor(regional_emulated_agb["lat"])),decimals=2)}')
    print(f'MSE value of CESM compared to Sothe: {torch.round(lat_adjusted_mse(torch.tensor(regional_sothe_soil["cSoilAbove1m"]),torch.tensor(regional_cesm_soil["cSoilAbove1m"]),torch.tensor(regional_emulated_agb["lat"])),decimals=2)}')
    print(f'MSE value of emulated compared to Sothe: {torch.round(lat_adjusted_mse(torch.tensor(regional_sothe_soil["cSoilAbove1m"]),torch.tensor(regional_emulated_soil["cSoilAbove1m"]),torch.tensor(regional_emulated_agb["lat"])),decimals=2)}')
    print(f'MSE value of emulated compared to CESM2: {torch.round(lat_adjusted_mse(torch.tensor(regional_cesm_agb["agb"]),torch.tensor(regional_emulated_agb["agb"]),torch.tensor(regional_emulated_agb["lat"])),decimals=2)}')
    print(f'MSE value of emulated compared to CESM2: {torch.round(lat_adjusted_mse(torch.tensor(regional_cesm_soil["cSoilAbove1m"]),torch.tensor(regional_emulated_soil["cSoilAbove1m"]),torch.tensor(regional_emulated_agb["lat"])),decimals=2)}')

if __name__ == "__main__":
    main()