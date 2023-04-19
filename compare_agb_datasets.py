import pandas as pd
import hydra 
from omegaconf import DictConfig
from preprocessing.utils import getArea,getGeometryBoxes
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

def getRegionalAGB(df:pd.DataFrame,ecozones_coords:pd.DataFrame,cfg):
    # df['lat'] = df['lat'].round(6)
    # # print(df['lat'].unique())
    # print(ecozones_coords['lat'].unique())
    df = pd.merge(df,ecozones_coords,on=['lat','lon'],how='inner')
    # print(df)
    df['area'] = df.apply(lambda x: getArea(x['lat'],x['lon'],cfg),axis=1)
    df['agb'] = df['agb'] * df['area'] /1e9 # make non-spatial,covnert to megatonnes
    #unit is now mtc 
    return df

def pandasToGeo(df:pd.DataFrame):
    boxes = getGeometryBoxes(df)
    df = gpd.GeoDataFrame(df['agb'], geometry=boxes)
    return df

#AGB FIGURE FOR ESTIMATE COMPARISON
def plotAGBComparison(dataframes:list,canada:gpd.GeoDataFrame,ecozones:gpd.GeoDataFrame,titles:list,filename:str) -> None:
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
    max_val = max([x['agb'].max() for x in dataframes])
    min_val = min([x['agb'].min() for x in dataframes])
    axes = axes.flatten()
    norm = mpl.colors.Normalize(min_val,max_val,clip=True)
    for ax_index in range(0,len(axes)):
        canada.plot(ax=axes[ax_index],alpha=0.1)
        ecozones.plot(ax=axes[ax_index],color='white',edgecolor='black',alpha=0.1)
        ax = dataframes[ax_index].plot(ax=axes[ax_index],column='agb',norm=norm,cmap='Greens')
        ax.set_xlabel('Longitude',fontsize=40)
        ax.set_ylabel('Latitude',fontsize=40)
        ax.tick_params(axis='both', which='major', labelsize=40)
        
        x = mpl.image.AxesImage(ax=axes[ax_index])
        axes[ax_index].title.set_text(titles[ax_index])
        axes[ax_index].title.set_fontsize(40)
    m = plt.cm.ScalarMappable(cmap='Greens')
    m.set_array(dataframes[0]['agb'])
    cbar = plt.colorbar(m,fraction=0.026, pad=0.04)
    cbar.ax.set_ylabel('AGB Carbon (Mt C)',fontsize=40)
    cbar.ax.tick_params(labelsize=40)

    f.tight_layout()
    f.suptitle('Comparison of Above-Ground Biomass Estimates',fontsize=60)
    plt.savefig(f'{filename}.png')

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    list_of_regions = ['Boreal Shield','Boreal Cordillera','Boreal PLain']

    #prepare data
    nfis_agb = pd.read_csv(f'{cfg.data}/nfis_agb_df.csv')
    walker_agb = pd.read_csv(f'{cfg.data}/walker_agb_df.csv')
    emulated = pd.read_csv(f'{cfg.data}/emulation_df.csv')
    cesm_data = pd.read_csv(f'{cfg.data}/cesm_data_variant.csv')
    cesm_data = cesm_data.groupby(['year','lat','lon']).mean().reset_index()     #transform from rolling window to yearly average
    emulated = emulated[emulated['year'] == 2015]
    cesm_data = cesm_data[cesm_data['year'] == 2014]
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

    #emulated has smallest area due to ERA nodata, so we need to merge on that
    walker_agb = pd.merge(walker_agb,emulated,on=['lat','lon'],how='inner',suffixes=('','_emulated'))
    nfis_agb = pd.merge(nfis_agb,emulated,on=['lat','lon'],how='inner',suffixes=('','_emulated'))
    cesm_data = pd.merge(cesm_data,emulated,on=['lat','lon'],how='inner',suffixes=('','_emulated'))

    #get regional AGB
    regional_nfis_agb = getRegionalAGB(nfis_agb,ecozones_coords,cfg)
    regional_walker_agb = getRegionalAGB(walker_agb,ecozones_coords,cfg)
    regional_emulated_agb = getRegionalAGB(emulated,ecozones_coords,cfg)
    regional_cesm_agb = getRegionalAGB(cesm_data,ecozones_coords,cfg)

    #shape files for plotting
    canada = gpd.read_file(f'{cfg.data}/shapefiles/lpr_000b16a_e/lpr_000b16a_e.shp')
    canada = canada.to_crs('4326')
    ecozones = gpd.read_file('data/shapefiles/ecozones.shp').to_crs('epsg:4326')
    ecozones = ecozones.where(ecozones['ZONE_NAME'].isin(['Boreal Shield','Boreal Cordillera','Boreal PLain']))

    #convert to geodataframes
    nfis_gdf = pandasToGeo(regional_nfis_agb)
    walker_gdf = pandasToGeo(regional_walker_agb)
    emulated_gdf = pandasToGeo(regional_emulated_agb)
    cesm_gdf = pandasToGeo(regional_cesm_agb)

    plotAGBComparison([nfis_gdf,walker_gdf,emulated_gdf,cesm_gdf],canada,ecozones,['Matasci et al. (2015)','Walker et al. (2015)','Emulated (2015)','CESM (2014)'],'agb_comparison')

    #REGIONAL SUMS
    for region in list_of_regions:
        print(f'For {region}')
        print(f'NFIS sum: {regional_nfis_agb[regional_nfis_agb["zone"]==region]["agb"].sum()}')
        print(f'Walker sum: {regional_walker_agb[regional_walker_agb["zone"]==region]["agb"].sum()}')
        print(f'Emulated sum: {regional_emulated_agb[regional_emulated_agb["zone"]==region]["agb"].sum()}')
        print(f'CESM sum: {regional_cesm_agb[regional_cesm_agb["zone"]==region]["agb"].sum()}')
    #TOTAL SUMS
    nfis_sum = round(regional_nfis_agb['agb'].sum(),2)
    walker_sum = round(regional_walker_agb['agb'].sum(),2)
    emulated_sum = round(regional_emulated_agb['agb'].sum(),2)
    cesm_sum = round(regional_cesm_agb['agb'].sum(),2)
    print(f'NFIS sum: {nfis_sum}')
    print(f'Walker sum: {walker_sum}')
    print(f'Emulated sum: {emulated_sum}')
    print(f'CESM sum: {cesm_sum}')

    #R2 value of emulation,CESM compared to NFIS, walker 
    print(f'R2 value of emulation compared to NFIS: {round(r2_score(regional_nfis_agb["agb"],regional_emulated_agb["agb"]),2)}')
    print(f'R2 value of emulation compared to Walker: {round(r2_score(regional_walker_agb["agb"],regional_emulated_agb["agb"]),2)}')
    print(f'R2 value of CESM compared to NFIS: {round(r2_score(regional_nfis_agb["agb"],regional_cesm_agb["agb"]), 2)}')
    print(f'R2 value of CESM compared to Walker: {round(r2_score(regional_walker_agb["agb"],regional_cesm_agb["agb"]), 2)}')

    #RMSE values of emulation,CESM compared to NFIS, walker
    print(f'RMSE value of emulation compared to NFIS: {round(np.sqrt(mean_squared_error(regional_nfis_agb["agb"],regional_emulated_agb["agb"])),2)}')
    print(f'RMSE value of emulation compared to Walker: {round(np.sqrt(mean_squared_error(regional_walker_agb["agb"],regional_emulated_agb["agb"])),2)}')
    print(f'RMSE value of CESM compared to NFIS: {round(np.sqrt(mean_squared_error(regional_nfis_agb["agb"],regional_cesm_agb["agb"])),2)}')
    print(f'RMSE value of CESM compared to Walker: {round(np.sqrt(mean_squared_error(regional_walker_agb["agb"],regional_cesm_agb["agb"])),2)}')

    
if __name__ == "__main__":
    main()