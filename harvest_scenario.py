import pandas as pd
from omegaconf import DictConfig
import hydra
import numpy as np
from infer_lstm import infer_lstm
from compare_agb_datasets import getRegionalAGB,pandasToGeo
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl


#AGB FIGURE FOR ESTIMATE COMPARISON
def plotAGBComparison(dataframes:list,canada:gpd.GeoDataFrame,ecozones:gpd.GeoDataFrame,titles:list,filename:str,main_title) -> None:
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
    # max_val = max([x['agb'].max() for x in dataframes])
    # min_val = min([x['agb'].min() for x in dataframes])
    axes = axes.flatten()
    for ax_index in range(0,len(axes)):
        max_val = dataframes[ax_index].agb.max()
        min_val = dataframes[ax_index].agb.min()
        norm = mpl.colors.Normalize(min_val,max_val,clip=True)

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
        m.set_array(dataframes[ax_index]['agb'])
        cbar = plt.colorbar(m,fraction=0.026, pad=0.04,ax=axes[ax_index])
        cbar.ax.set_ylabel('AGB Carbon (Mt C)',fontsize=40)
        cbar.ax.tick_params(labelsize=40)

    f.tight_layout()
    f.suptitle(main_title,fontsize=60)
    plt.savefig(f'{filename}.png')


def yearlyComparisonPlot(dataframes:list,legend:list):
    fig,ax = plt.subplots(figsize=(20,10))
    for df in dataframes:
        df.plot(x='year',y='agb',ax=ax)
    ax.set_xlabel('Year',fontsize=30)
    ax.set_ylabel('AGB Mt C',fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.legend(legend)
    plt.title('Yearly Above-Ground Biomass Totals for Harvest Scenario',fontsize=40)
    plt.savefig('yearly_comparison_harvest.png')
    
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    #make harvest datasets
    observed_ds = pd.read_csv(f'{cfg.data}/observed_timeseries{cfg.model.seq_len}_data.csv')
    list_of_regions = ['Boreal Shield','Boreal Cordillera','Boreal PLain']

    nfis_data = pd.read_csv(f'{cfg.data}/forest_df.csv')

    nfis_data['lat'] = nfis_data['lat'].round(6)
    observed_ds['lat'] = observed_ds['lat'].round(6)
    df_merged = pd.merge(observed_ds,nfis_data,on=['year','lat','lon'],how='left')
    emulated = pd.read_csv(f'{cfg.data}/emulation_df.csv')
    ecozones_coords = pd.read_csv(f'{cfg.data}/ecozones_coordinates.csv')
    ecozones_coords = ecozones_coords[ecozones_coords['zone'].isin(list_of_regions)]
    ecozones_coords['lat'] = ecozones_coords['lat'].round(6)
    df_merged = pd.merge(df_merged,ecozones_coords,on=['lat','lon'],how='inner')

    harvest_yearly_data = pd.merge(df_merged,df_merged.pivot_table(index=['lat','lon'], 
                columns=['year'], values='percent_harvested').reset_index(),on=['lat','lon'],how='left')
    growth_yearly_data = pd.merge(df_merged,df_merged.pivot_table(index=['lat','lon'], 
                columns=['year'], values='percentage_growth').reset_index(),on=['lat','lon'],how='left')


    for year in range(1985,2020):
        harvest_yearly_data.loc[harvest_yearly_data['year'] > year,'treeFrac'] = (harvest_yearly_data.loc[harvest_yearly_data['year'] > year][year] * 100) + harvest_yearly_data.loc[harvest_yearly_data['year'] > year,'treeFrac']
        growth_yearly_data.loc[growth_yearly_data['year'] > year,'treeFrac'] = growth_yearly_data.loc[growth_yearly_data['year'] > year,'treeFrac']  - (growth_yearly_data.loc[growth_yearly_data['year'] > year][year] * 100)


    #infer using model
    reforested_carbon = infer_lstm(harvest_yearly_data,cfg)
    no_regrowth_carbon = infer_lstm(growth_yearly_data,cfg)



    emulated['agb'] = emulated['cStem'] + emulated['cOther'] + emulated['cLeaf'] #this is in kg/m2
    reforested_carbon['agb'] = reforested_carbon['cStem'] + reforested_carbon['cOther'] + reforested_carbon['cLeaf'] #this is in kg/m2
    no_regrowth_carbon['agb'] = no_regrowth_carbon['cStem'] + no_regrowth_carbon['cOther'] + no_regrowth_carbon['cLeaf'] #this is in kg/m2


    regional_reforested_agb = getRegionalAGB(reforested_carbon,ecozones_coords,cfg)
    regional_no_regrowth_agb = getRegionalAGB(no_regrowth_carbon,ecozones_coords,cfg)
    regional_emulated_agb = getRegionalAGB(emulated,ecozones_coords,cfg)
    regional_no_regrowth_difference = regional_reforested_agb.copy()
    regional_reforested_difference = regional_reforested_agb.copy()
    regional_no_regrowth_difference['agb'] = regional_emulated_agb['agb'] - regional_no_regrowth_agb['agb']
    regional_reforested_difference['agb'] = regional_reforested_agb['agb'] - regional_emulated_agb['agb']

    #shape files for plotting
    canada = gpd.read_file(f'{cfg.data}/shapefiles/lpr_000b16a_e/lpr_000b16a_e.shp')
    canada = canada.to_crs('4326')
    ecozones = gpd.read_file('data/shapefiles/ecozones.shp').to_crs('epsg:4326')
    ecozones = ecozones.where(ecozones['ZONE_NAME'].isin(['Boreal Shield','Boreal Cordillera','Boreal PLain']))

    #convert to geodataframes
    reforested_gdf = pandasToGeo(regional_reforested_agb)
    no_regrowth_gdf = pandasToGeo(regional_no_regrowth_agb)
    emulated_gdf = pandasToGeo(regional_emulated_agb)

    no_regrowth_diff = pandasToGeo(regional_no_regrowth_difference)
    reforested_diff = pandasToGeo(regional_reforested_difference)
    plotAGBComparison([emulated_gdf,no_regrowth_gdf,no_regrowth_diff],canada,ecozones,['Observed','No Regrowth','Difference'],'regrowth_harvest_scenario','No Regrowth Harvest Scenario')
    plotAGBComparison([reforested_gdf,emulated_gdf,reforested_diff],canada,ecozones,['Reforestation','Observed','Difference'],'reforest_harvest_scenario','Reforestation Harvest Scenario')

    #plot yearly carbon
    reforested_carbon_yearly = regional_reforested_agb.groupby(['year']).sum().reset_index()
    no_regrowth_carbon_yearly = regional_no_regrowth_agb.groupby(['year']).sum().reset_index()
    emulated_yearly = regional_emulated_agb.groupby(['year']).sum().reset_index()
    yearlyComparisonPlot([emulated_yearly,reforested_carbon_yearly,no_regrowth_carbon_yearly],['Observed','Reforested','No Regrowth'])

    #zones for year 2015 
    regional_reforested_agb_2015 = regional_reforested_agb[regional_reforested_agb['year'] == 2015].groupby('zone').sum().reset_index()
    regional_no_regrowth_agb_2015 = regional_no_regrowth_agb[regional_no_regrowth_agb['year'] == 2015].groupby('zone').sum().reset_index()
    regional_emulated_agb_2015 = regional_emulated_agb[regional_emulated_agb['year'] == 2015].groupby('zone').sum().reset_index()

    print(regional_reforested_agb_2015)
    print(regional_no_regrowth_agb_2015)
    print(regional_emulated_agb_2015)
    print(reforested_carbon_yearly)
    print(no_regrowth_carbon_yearly)
    print(emulated_yearly)

if __name__ == "__main__":
    main()
