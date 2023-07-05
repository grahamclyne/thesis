import pandas as pd
from omegaconf import DictConfig
import hydra
import numpy as np
from analysis.infer_lstm import infer_lstm
from analysis.compare_observed_datasets import getRegionalValues,pandasToGeo
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl


def plotHarvestComparison(dataframes:list,canada:gpd.GeoDataFrame,ecozones:gpd.GeoDataFrame,titles:list,filename:str,main_title,column:str) -> None:
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
    max_val = max([x[column].max() for x in dataframes])
    min_val = min([x[column].min() for x in dataframes])
    min_val = -100
    max_val = 1000
    axes = axes.flatten()
    norm = mpl.colors.SymLogNorm(1,vmin=min_val,vmax=max_val)
    # norm = mpl.colors.LogNorm(vmin=min_val,vmax=max_val)
    for ax_index in range(0,len(axes)):
        # max_val = dataframes[ax_index][column].max()
        # min_val = dataframes[ax_index][column].min()
        # norm = mpl.colors.Normalize(min_val,max_val,clip=True)

        canada.plot(ax=axes[ax_index],alpha=0.1)
        ecozones.plot(ax=axes[ax_index],color='white',edgecolor='black',alpha=0.1)
        ax = dataframes[ax_index].plot(ax=axes[ax_index],column=column,norm=norm,cmap='Greens')
        ax.set_xlabel('Longitude',fontsize=40)
        ax.set_ylabel('Latitude',fontsize=40)
        ax.tick_params(axis='both', which='major', labelsize=40)
        
        x = mpl.image.AxesImage(ax=axes[ax_index])
        axes[ax_index].title.set_text(titles[ax_index])
        axes[ax_index].title.set_fontsize(40)

    m = plt.cm.ScalarMappable(cmap='Greens',norm=norm)
    # m.set_array(dataframes[0][column])
    m.set_array(np.linspace(min_val,max_val,100))

    cbar = plt.colorbar(m,fraction=0.026, pad=0.04,ax=axes[ax_index])
    cbar.ax.set_ylabel(f'{column} (kg/m2)',fontsize=40)
    cbar.ax.tick_params(labelsize=30)

    f.tight_layout()
    # f.suptitle(main_title,fontsize=60)
    plt.savefig(f'figures/{filename}.png')


def yearlyComparisonPlot(dataframes:list,legend:list,column:str):
    fig,ax = plt.subplots(figsize=(20,10))
    for df_index in range(len(dataframes)):
        dataframes[df_index].plot(x='year',y=column,ax=ax)
    ax.set_xlabel('Year',fontsize=30)
    ax.set_ylabel(f'{column} Mt C',fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.legend(legend,fontsize=30)
    # plt.title('Yearly Above-Ground Biomass Totals for Harvest Scenario',fontsize=40)
    plt.savefig(f'figures/yearly_comparison_harvest_{column}.png')
    
@hydra.main(version_base=None, config_path="../conf", config_name="config")
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

    print(harvest_yearly_data['treeFrac'].describe())






    #infer using model
    reforested_carbon = infer_lstm(harvest_yearly_data,cfg)
    no_regrowth_carbon = infer_lstm(growth_yearly_data,cfg)



    emulated['agb'] = emulated['cStem'] + emulated['cOther'] + emulated['cLeaf'] #this is in kg/m2
    reforested_carbon['agb'] = reforested_carbon['cStem'] + reforested_carbon['cOther'] + reforested_carbon['cLeaf'] #this is in kg/m2
    no_regrowth_carbon['agb'] = no_regrowth_carbon['cStem'] + no_regrowth_carbon['cOther'] + no_regrowth_carbon['cLeaf'] #this is in kg/m2


    regional_reforested_agb = getRegionalValues(reforested_carbon,ecozones_coords,cfg,'agb')
    regional_no_regrowth_agb = getRegionalValues(no_regrowth_carbon,ecozones_coords,cfg,'agb')
    regional_emulated_agb = getRegionalValues(emulated,ecozones_coords,cfg,'agb')

    regional_reforested_soil = getRegionalValues(reforested_carbon,ecozones_coords,cfg,'cSoilAbove1m')
    regional_no_regrowth_soil = getRegionalValues(no_regrowth_carbon,ecozones_coords,cfg,'cSoilAbove1m')
    regional_emulated_soil = getRegionalValues(emulated,ecozones_coords,cfg,'cSoilAbove1m')



    regional_no_regrowth_difference = regional_reforested_agb.copy()
    regional_reforested_difference = regional_reforested_agb.copy()
    regional_no_regrowth_difference['agb'] = regional_emulated_agb['agb'] - regional_no_regrowth_agb['agb']
    regional_reforested_difference['agb'] = regional_reforested_agb['agb'] - regional_emulated_agb['agb']

    regional_no_regrowth_soil_difference = regional_reforested_soil.copy()
    regional_reforested_soil_difference = regional_reforested_soil.copy()
    regional_no_regrowth_soil_difference['cSoilAbove1m'] = regional_emulated_soil['cSoilAbove1m'] - regional_no_regrowth_soil['cSoilAbove1m']
    regional_reforested_soil_difference['cSoilAbove1m'] = regional_reforested_soil['cSoilAbove1m'] - regional_emulated_soil['cSoilAbove1m']



    #shape files for plotting
    canada = gpd.read_file(f'{cfg.data}/shapefiles/lpr_000b16a_e/lpr_000b16a_e.shp')
    canada = canada.to_crs('4326')
    ecozones = gpd.read_file('data/shapefiles/ecozones.shp').to_crs('epsg:4326')
    ecozones = ecozones.where(ecozones['ZONE_NAME'].isin(['Boreal Shield','Boreal Cordillera','Boreal PLain']))









    #convert to geodataframes
    reforested_gdf = pandasToGeo(regional_reforested_agb,'agb')
    no_regrowth_gdf = pandasToGeo(regional_no_regrowth_agb,'agb')
    emulated_gdf = pandasToGeo(regional_emulated_agb,'agb')
    no_regrowth_diff = pandasToGeo(regional_no_regrowth_difference,'agb')
    reforested_diff = pandasToGeo(regional_reforested_difference,'agb')

    reforested_gdf_soil = pandasToGeo(regional_reforested_soil,'cSoilAbove1m')
    no_regrowth_gdf_soil = pandasToGeo(regional_no_regrowth_soil,'cSoilAbove1m')
    emulated_gdf_soil = pandasToGeo(regional_emulated_soil,'cSoilAbove1m')
    no_regrowth_diff_soil = pandasToGeo(regional_no_regrowth_soil_difference,'cSoilAbove1m')
    reforested_diff_soil = pandasToGeo(regional_reforested_soil_difference,'cSoilAbove1m')

    # # Plotting ---

    regional_reforested_treeFrac = getRegionalValues(harvest_yearly_data,ecozones_coords,cfg,'treeFrac')
    regional_no_regrowth_treeFrac = getRegionalValues(growth_yearly_data,ecozones_coords,cfg,'treeFrac')
    nfis_tree_cover = getRegionalValues(df_merged,ecozones_coords,cfg,'treeFrac')
    nfis_tree_cover_geo = pandasToGeo(nfis_tree_cover,'treeFrac')
    tree_frac_geo = pandasToGeo(regional_reforested_treeFrac,'treeFrac')
    no_regrowth_geo = pandasToGeo(regional_no_regrowth_treeFrac,'treeFrac')

    # print(tree_frac_geo['treeFrac'].describe())
    # print(nfis_tree_cover_geo['treeFrac'].describe())

    tree_frac_geo['treeFrac'] = tree_frac_geo['treeFrac'] - nfis_tree_cover_geo['treeFrac']
    no_regrowth_geo['treeFrac'] = no_regrowth_geo['treeFrac'] - nfis_tree_cover_geo['treeFrac']

    def make_gif(df,title,file_name):
        import matplotlib.pyplot as plt
        import seaborn
        from celluloid import Camera
        seaborn.set_context("paper")
        fig, ax1 = plt.subplots(1)
        camera = Camera(fig)
        # ax1.set_xticks([], [])
        max_val = df['treeFrac'].max()
        min_val = df['treeFrac'].min()
        norm = mpl.colors.SymLogNorm(1,vmin=min_val,vmax=max_val)
        m = plt.cm.ScalarMappable(cmap='Greens',norm=norm)
        m.set_array(np.linspace(min_val,max_val,100))
        for year_id in range(1986,2020):
            # ax1.text(0.00, 1.07, f'{title} {year_id}', transform=ax1.transAxes,fontsize=10)
            canada.plot(ax=ax1,color='white',edgecolor='black',alpha=0.1)
            ecozones.plot(ax=ax1,color='white',edgecolor='black',alpha=0.1)
            tree_frac_g = df[df['year']==year_id]
            tree_frac_g.plot(ax=ax1,column='treeFrac',norm=norm,cmap='Greens')
            ax1.set_ylabel('Longitude',fontsize=10)
            ax1.set_xlabel('Latitude',fontsize=10)
            camera.snap()
        cbar = plt.colorbar(m,fraction=0.026, pad=0.04)
        cbar.ax.set_ylabel(f'% change',fontsize=10)
        anim = camera.animate()
        anim.save(f"figures/{file_name}.gif", dpi=150, writer="imagemagick")

    # make_gif(tree_frac_geo,'Tree Cover Percentage Change From Reforest Scenario' ,'regrowth')
    # make_gif(no_regrowth_geo,'Tree Cover Percentage Change From No Regrowth Scenario' ,'no_regrowth')

    plotHarvestComparison([emulated_gdf,no_regrowth_gdf,no_regrowth_diff],canada,ecozones,['ERANFIS','No Regrowth','Difference'],'regrowth_harvest_scenario','No Regrowth Harvest Scenario','agb')
    plotHarvestComparison([reforested_gdf,emulated_gdf,reforested_diff],canada,ecozones,['Reforestation','ERANFIS','Difference'],'reforest_harvest_scenario','Reforestation Harvest Scenario','agb')

    plotHarvestComparison([emulated_gdf_soil,no_regrowth_gdf_soil,no_regrowth_diff_soil],canada,ecozones,['ERANFIS','No Regrowth','Difference'],'regrowth_soil_harvest_scenario','No Regrowth Harvest Scenario','cSoilAbove1m')
    plotHarvestComparison([reforested_gdf_soil,emulated_gdf_soil,reforested_diff_soil],canada,ecozones,['Reforestation','ERANFIS','Difference'],'reforest_soil_harvest_scenario','Reforestation Harvest Scenario','cSoilAbove1m')

    #plot yearly carbon
    reforested_carbon_yearly = regional_reforested_agb.groupby(['year']).sum().reset_index()
    no_regrowth_carbon_yearly = regional_no_regrowth_agb.groupby(['year']).sum().reset_index()
    emulated_yearly = regional_emulated_agb.groupby(['year']).sum().reset_index()
    yearlyComparisonPlot([emulated_yearly,reforested_carbon_yearly,no_regrowth_carbon_yearly],['AGB ERANFIS','AGB Reforested','AGB No Regrowth'],'agb')

    #plot yearly soil
    reforested_soil_yearly = regional_reforested_soil.groupby(['year']).sum().reset_index()
    no_regrowth_soil_yearly = regional_no_regrowth_soil.groupby(['year']).sum().reset_index()
    emulated_soil_yearly = regional_emulated_soil.groupby(['year']).sum().reset_index()
    yearlyComparisonPlot([emulated_soil_yearly,reforested_soil_yearly,no_regrowth_soil_yearly],['Soil ERANFIS','Soil Reforested','Soil No Regrowth'],'cSoilAbove1m')


    #zones for year 2019
    regional_reforested_agb_2019 = regional_reforested_agb[regional_reforested_agb['year'] == 2019].groupby('zone').sum().reset_index()
    regional_no_regrowth_agb_2019 = regional_no_regrowth_agb[regional_no_regrowth_agb['year'] == 2019].groupby('zone').sum().reset_index()
    regional_emulated_agb_2019 = regional_emulated_agb[regional_emulated_agb['year'] == 2019].groupby('zone').sum().reset_index()

    regional_reforested_soil_2019 = regional_reforested_soil[regional_reforested_soil['year'] == 2019].groupby('zone').sum().reset_index()
    regional_no_regrowth_soil_2019 = regional_no_regrowth_soil[regional_no_regrowth_soil['year'] == 2019].groupby('zone').sum().reset_index()
    regional_emulated_soil_2019 = regional_emulated_soil[regional_emulated_soil['year'] == 2019].groupby('zone').sum().reset_index()


    print(regional_reforested_agb_2019)
    print(regional_no_regrowth_agb_2019)
    print(regional_emulated_agb_2019)
    print(regional_reforested_soil_2019)
    print(regional_no_regrowth_soil_2019)
    print(regional_emulated_soil_2019)
    print(reforested_carbon_yearly)
    print(no_regrowth_carbon_yearly)
    print(emulated_yearly)
    print(reforested_soil_yearly)
    print(no_regrowth_soil_yearly)
    print(emulated_soil_yearly)
if __name__ == "__main__":
    main()
