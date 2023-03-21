import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra
import wandb
from infer_lstm import infer_lstm
import geopandas as gpd
from preprocessing.utils import getGeometryBoxes,getArea
import matplotlib as mpl
from visualization.plot_helpers import plotCountryWideGridded



def plotComparison(df1:pd.DataFrame,df2:pd.DataFrame,df3:pd.DataFrame,variable:str):
    df2['id'] = 'reforestation'
    df3['id'] = 'no reforestation'
    df1['id'] = 'CESM'
    df_final = pd.concat([df1,df2,df3]).groupby(['year','id']).sum().reset_index()
    table = wandb.Table(data=df_final[[variable,'year','id']])
    wandb.log({variable + '_df' : table})
    fig, ax = plt.subplots()
    ax.plot(df1.groupby('year').sum()[variable],label='CESM')
    ax.plot(df2.groupby('year').sum()[variable],label='Reforestation')
    ax.plot(df3.groupby('year').sum()[variable],label='No Reforestation')
    ax.legend()
    ax.title.set_text(f'{variable}')
    # ax.axhline(0, color='black', linewidth=.5)
    # wandb.log({f'{variable}_CESM':wandb.Image(fig)})

    fig.savefig(f'{variable}_lstm.png')

def plotDifference(reforest:pd.DataFrame,no_reforest:pd.DataFrame):
    agb_diff = reforest.groupby('year').sum()['agb'] - no_reforest.groupby('year').sum()['agb']
    soil_diff = reforest.groupby('year').sum()['cSoilAbove1m']- no_reforest.groupby('year').sum()['cSoilAbove1m']
    print(agb_diff)
    print(soil_diff)
    fig, ax = plt.subplots()
    ax.plot(agb_diff,label='Above-ground Biomass')
    ax.plot(soil_diff,label='Soil Carbon Above 1m')
    ax.set_ylabel('Mt Carbon',fontsize=50)
    ax.set_xlabel('year',fontsize=50)
    ax.legend()

    ax.title.set_text(f'Gains from Reforestation Scenario')
    # ax.axhline(0, color='black', linewidth=.5)
    # wandb.log({f'{variable}_CESM':wandb.Image(fig)})

    fig.savefig(f'Difference_lstm.png')


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cesm_data = pd.read_csv('data/cesm_data_variant.csv')

    cesm_data = cesm_data[cfg.model.input + cfg.model.output + cfg.model.id]
    cesm_data = cesm_data[cesm_data.year > 2012]
    cesm_data = cesm_data.groupby(['year','lat','lon']).mean().reset_index()
    # cesm_data = cesm_data.assign(lat=round(cesm_data['lat'],6))
    # cesm_data = cesm_data[cesm_data.lat != 42.879582]
    observed_input = pd.read_csv(f'{cfg.data}/observed_timeseries30_data.csv')
    # observed_input = observed_input[['year','lat','lon','treeFrac']]
    reforested_input = pd.read_csv(f'{cfg.data}/observed_reforest_ts.csv')
    reforested_input['lat'] = round(reforested_input['lat'],6)
    observed_input['lat'] = round(observed_input['lat'],6)
    reforested_input.rename(columns={'year_x':'year'},inplace=True)
    # reforested_input = reforested_input[reforested_input['year'] < 2016]
    # observed_input = observed_input[observed_input['year'] < 2016]

    # reforested_input.fillna(reforested_input.median(),inplace=True)
    # reforested_input.fillna(reforested_input.median(),inplace=True)
    # reforested_input = reforested_input.where((reforested_input['year'] > 1984) & (reforested_input['year'] < 2015)).dropna()
    # reforested_input = reforested_input[['year','lat','lon','treeFrac']]
    # print(observed_input.where(observed_input['treeFrac']))
    # df_merged = pd.merge(cesm_data,observed_input,on=['year','lat','lon'],how='left')
    # reforested_merged = pd.merge(cesm_data,reforested_input,on=['year','lat','lon'],how='left')
    # df_merged = df_merged.drop(columns=['treeFrac_x'])
    # df_merged = df_merged.rename(columns={'treeFrac_y':'treeFrac'})
    # pd.set_option('display. max_rows', None)
    # empty_coords = df_merged[df_merged['treeFrac'].isna()].dropna(how='all')[['lat','lon']].drop_duplicates().reset_index()
    # print(empty_coords)
    # print(df_merged.where(df_merged['lat'].isin(empty_coords['lat'])))
    # df_merged = df_merged.where(~((df_merged['lat'].isin(empty_coords['lat'])) & (df_merged['lon'].isin(empty_coords['lon'])))).dropna()
    # cesm_data = cesm_data.where(~((cesm_data['lat'].isin(empty_coords['lat'])) & (cesm_data['lon'].isin(empty_coords['lon'])))).dropna()
    # reforested_merged = reforested_merged.where(~((reforested_merged['lat'].isin(empty_coords['lat'])) & (reforested_merged['lon'].isin(empty_coords['lon'])))).dropna()



    # reforested_merged = reforested_merged.drop(columns=['treeFrac_x'])
    # reforested_merged = reforested_merged.rename(columns={'treeFrac_y':'treeFrac'})
    # df_merged.interpolate(method='spline',order=5,inplace=True)
    # reforested_merged.interpolate(method='spline',order=5,inplace=True)
    # reforested_merged.dropna(how='any',inplace=True)
    no_reforest_infer = infer_lstm(observed_input,cfg)
    no_reforest_infer.to_csv('data/lstm_hybrid_no_reforest.csv')
    reforest_infer = infer_lstm(reforested_input,cfg)
    reforest_infer = reforest_infer[reforest_infer['year'] == 2014]
    no_reforest_infer = no_reforest_infer[no_reforest_infer['year'] == 2014]
    # reforest_infer.to_csv('data/lstm_hybrid_reforest.csv')
    # cesm_data = cesm_data[29::30].groupby(['year','lat','lon']).sum().reset_index()

    # cesm_data = cesm_data[~((cesm_data.lat == 68.324608) & (cesm_data.lon == -136.25))]
    # print(cesm_data.describe())
    # print(no_reforest_infer.describe())
    # for var in cfg.model.output:
    #     if(var == "cSoilAbove1m"):
    #         continue
    #     reforest_infer.loc[:,var] = reforest_infer.loc[:,var] / 1000000000
    #     no_reforest_infer.loc[:,var] = no_reforest_infer.loc[:,var] / 1000000000

    reforested_input = reforested_input[29::30].groupby(['year','lat','lon']).sum().reset_index()
    observed_input = observed_input[29::30].groupby(['year','lat','lon']).sum().reset_index()
    observed_input = observed_input[observed_input['year'] == 2014]
    reforested_input = reforested_input[reforested_input['year'] == 2014]

    cesm_data['agb'] = cesm_data['cStem'] + cesm_data['cLeaf'] + cesm_data['cOther']
    reforest_infer['agb'] = reforest_infer['cStem'] + reforest_infer['cLeaf'] + reforest_infer['cOther']
    no_reforest_infer['agb'] = no_reforest_infer['cStem'] + no_reforest_infer['cLeaf'] + no_reforest_infer['cOther']
    reforest_infer['area'] = reforest_infer.apply(lambda x: getArea(x['lat'],x['lon']),axis=1)
    reforest_infer['agb'] = reforest_infer['agb'] * reforest_infer['area'] / 1e9
    reforest_infer['cSoilAbove1m'] = reforest_infer['cSoilAbove1m'] * reforest_infer['area'] / 1e9

    no_reforest_infer['area'] = reforest_infer.apply(lambda x: getArea(x['lat'],x['lon']),axis=1)
    no_reforest_infer['agb'] = no_reforest_infer['agb'] * no_reforest_infer['area'] / 1e9
    no_reforest_infer['cSoilAbove1m'] = no_reforest_infer['cSoilAbove1m'] * no_reforest_infer['area'] / 1e9
  
    no_reforest_infer.to_csv('no_reforest_infer.csv')
    reforest_infer.to_csv('reforest_infer.csv')
    observed_input.to_csv('observed_input.csv')
    reforested_input.to_csv('reforested_input.csv')
    #plot agb of reforested vs non reforested
    subplot_titles = ['Simulated Reforestation','Observed Forest Cover','Difference']
    plotCountryWideGridded([gpd.GeoDataFrame(reforest_infer['agb'],geometry=getGeometryBoxes(reforest_infer)),
                            gpd.GeoDataFrame(no_reforest_infer['agb'],geometry=getGeometryBoxes(no_reforest_infer)),
                            gpd.GeoDataFrame(reforest_infer['agb'] - no_reforest_infer['agb'],geometry=getGeometryBoxes(reforest_infer))
    ],'agb',subplot_titles,'Comparison of Above-Ground Biomass for Simulated Reforestation')

    #plot soil of reforested vs non reforested
    plotCountryWideGridded([gpd.GeoDataFrame(reforest_infer['cSoilAbove1m'],geometry=getGeometryBoxes(reforest_infer)),
                            gpd.GeoDataFrame(no_reforest_infer['cSoilAbove1m'],geometry=getGeometryBoxes(no_reforest_infer)),
                            gpd.GeoDataFrame(reforest_infer['cSoilAbove1m'] - no_reforest_infer['cSoilAbove1m'],geometry=getGeometryBoxes(reforest_infer))
    ],'cSoilAbove1m',subplot_titles,'Comparison of Soil Above 1m Depth for Simulated Reforestation')


    plotCountryWideGridded([gpd.GeoDataFrame(reforested_input['treeFrac'],geometry=getGeometryBoxes(reforested_input)),
                            gpd.GeoDataFrame(observed_input['treeFrac'],geometry=getGeometryBoxes(observed_input)),
                            gpd.GeoDataFrame(reforested_input['treeFrac'] - observed_input['treeFrac'],geometry=getGeometryBoxes(reforested_input))
    ],'treeFrac',subplot_titles,'Comparison of Tree Cover for Simulated Reforestation')
main()