import pandas as pd
from omegaconf import DictConfig
import hydra
import numpy as np
from infer_lstm import infer_lstm
from compare_agb_datasets import getRegionalAGB,plotAGBComparison,pandasToGeo
import geopandas as gpd
import matplotlib.pyplot as plt

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

    nfis_data = pd.read_csv(f'{cfg.data}/forest_df.csv')

    nfis_data['lat'] = nfis_data['lat'].round(6)
    observed_ds['lat'] = observed_ds['lat'].round(6)
    df_merged = pd.merge(observed_ds,nfis_data,on=['year','lat','lon'],how='left')
    emulated = pd.read_csv(f'{cfg.data}/emulation_df.csv')

    # #convert from rolling window data
    # emulated = emulated.groupby(['lat','lon','year']).mean().reset_index()
    df_merged['without_regrowth'] = df_merged['treeFrac'] - (df_merged['percentage_growth'] * 100)
    df_merged['reforested'] = df_merged['treeFrac'] +  (df_merged['percent_harvested'] * 100)
    print(df_merged[['reforested','without_regrowth','treeFrac','percent_harvested','percentage_growth']])
    df_merged.drop(columns=['treeFrac'],inplace=True)
    reforestation = df_merged.rename(columns={'reforested':'treeFrac'})
    no_regrowth_df = df_merged.rename(columns={'without_regrowth':'treeFrac'})


    #infer using model
    reforested_carbon = infer_lstm(reforestation,cfg)
    no_regrowth_carbon = infer_lstm(no_regrowth_df,cfg)

   

    print(reforested_carbon)
    print(no_regrowth_carbon)
    

    list_of_regions = ['Boreal Shield','Boreal Cordillera','Boreal PLain']

    ecozones_coords = pd.read_csv(f'{cfg.data}/ecozones_coordinates.csv')
    ecozones_coords = ecozones_coords[ecozones_coords['zone'].isin(list_of_regions)]
    ecozones_coords['lat'] = ecozones_coords['lat'].round(6)
    emulated['agb'] = emulated['cStem'] + emulated['cOther'] + emulated['cLeaf'] #this is in kg/m2
    reforested_carbon['agb'] = reforested_carbon['cStem'] + reforested_carbon['cOther'] + reforested_carbon['cLeaf'] #this is in kg/m2
    no_regrowth_carbon['agb'] = no_regrowth_carbon['cStem'] + no_regrowth_carbon['cOther'] + no_regrowth_carbon['cLeaf'] #this is in kg/m2


    regional_reforested_agb = getRegionalAGB(reforested_carbon,ecozones_coords,cfg)
    regional_no_regrowth_agb = getRegionalAGB(no_regrowth_carbon,ecozones_coords,cfg)
    regional_emulated_agb = getRegionalAGB(emulated,ecozones_coords,cfg)
    regional_no_regrowth_difference = regional_reforested_agb.copy()
    regional_reforested_difference = regional_reforested_agb.copy()
    regional_no_regrowth_difference['agb'] = regional_no_regrowth_agb['agb'] - regional_emulated_agb['agb']
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
    plotAGBComparison([no_regrowth_gdf,emulated_gdf,no_regrowth_diff],canada,ecozones,['No Regrowth','Observed','Difference'],'regrowth_harvest_scenario')
    plotAGBComparison([reforested_gdf,emulated_gdf,reforested_diff],canada,ecozones,['Reforestation','Observed','Difference'],'reforest_harvest_scenario')

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
