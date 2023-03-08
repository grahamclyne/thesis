#INFER MODEL W HYBRID DATA 
import pandas as pd
from pickle import load 
from lstm_model import RegressionLSTM
from transformer.transformer_model import CMIPTimeSeriesDataset
import numpy as np
import torch as T 
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra
import numpy as np    
from preprocessing.utils import getArea
import wandb
import omegaconf


def plotTreeFracComparison(df1,df2):
    fig, ax = plt.subplots()
    # df1['area'] =  df1.apply(lambda x: getArea(x['lat'],x['lon']),axis=1)
    # df2['area'] =  df2.apply(lambda x: getArea(x['lat'],x['lon']),axis=1)
    # df1['treeFrac'] = df1['treeFrac']/100*df1['area']
    x1 = df1.groupby('year').mean()['treeFrac']/100 
    x2 = df2.groupby('year').mean()['treeFrac']/100
    print(x1)
    print(x2)
    ax.plot(x1,label='Reforested tree Fraction')
    ax.plot(x2,label='Not reforestation tree Fraction')
    ax.legend()
    ax.title.set_text(f'Tree Fraction')
    # ax.axhline(0, color='black', linewidth=.5)
    # wandb.log({f'{variable}_CESM':wandb.Image(fig)})

    fig.savefig(f'treeFrac_lstm.png')



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

def infer_lstm(data,cfg):
    model_name = cfg.run_name
    final_input = data[cfg.model.input +cfg.model.id]
    scaler = load(open(f'{cfg.project}/checkpoint/lstm_scaler_{model_name}.pkl', 'rb'))
    # out_scaler = load(open(f'{cfg.project}/checkpoint/lstm_output_scaler.pkl', 'rb'))
    # scaler = StandardScaler()
    # final_input['pr'] = final_input['pr'].apply(lambda x: x*10.4)

    final_input.loc[:,cfg.model.input ] =scaler.transform(final_input.loc[:,cfg.model.input])

    # final_input[cfg.model.input]=final_input[cfg.model.input].apply(scaler.transform)
    # hold_out_out_scaler = StandardScaler().fit(final_input[29::30].loc[:,cfg.model.output])
    model = RegressionLSTM(num_sensors=len(cfg.model.input), hidden_units=cfg.model.hidden_units,cfg=cfg)
    checkpoint = T.load(f'{cfg.project}/checkpoint/lstm_checkpoint_{model_name}.pt')

    #in case distributed training was used
    for key in list(checkpoint['model_state_dict'].keys()):
        checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'].pop(key)



    model.load_state_dict(checkpoint['model_state_dict'])
    final_input = final_input[cfg.model.input + cfg.model.id]
    ds = CMIPTimeSeriesDataset(final_input,cfg.model.seq_len,len(cfg.model.input) + len(cfg.model.id),cfg)
    batch_size = 1
    ldr = T.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=False)
    results = []
    ids = []
    # tgts = []
    model.eval()
    for X,tgt,id in ldr:
        y = model(X.float())
        y = (y.detach().numpy())
        results.extend(y)
        ids.append(id.detach().numpy())
        # tgts.extend(tgt.detach().numpy())
        # print(y)
        # print(tgt)
        # print(id)
    ids = np.concatenate(ids)
    results_df = pd.DataFrame(results)
    # tgts_df = pd.DataFrame(tgts)
    # tgts_df['year'] = ids[:,0]
    # tgts_df['lat'] = ids[:,1]
    # tgts_df['lon'] = ids[:,2]
    results_df['year'] = ids[:,0]
    results_df['lat'] = ids[:,1]
    results_df['lon'] = ids[:,2]
    # tgts_df.columns = cfg.model.output + ['year','lat','lon']
    results_df.columns = cfg.model.output + ['year','lat','lon']
    # print(results_df)   
    return results_df


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    wandb.init(project="inference", entity="gclyne",config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    cesm_data = pd.read_csv('data/cesm_data_variant.csv')
    # print(cesm_data.head())
    # print(cesm_data.year.unique())
    cesm_data = cesm_data[cfg.model.input + cfg.model.output + cfg.model.id]
    cesm_data = cesm_data[cesm_data.year > 2012]
    cesm_data = cesm_data.groupby(['year','lat','lon']).mean().reset_index()
    # cesm_data = cesm_data.assign(lat=round(cesm_data['lat'],6))
    # cesm_data = cesm_data[cesm_data.lat != 42.879582]
    print(cesm_data.head())
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
    print(observed_input.describe())
    print(reforested_input.describe())




    # reforested_merged = reforested_merged.drop(columns=['treeFrac_x'])
    # reforested_merged = reforested_merged.rename(columns={'treeFrac_y':'treeFrac'})
    # df_merged.interpolate(method='spline',order=5,inplace=True)
    # reforested_merged.interpolate(method='spline',order=5,inplace=True)
    # reforested_merged.dropna(how='any',inplace=True)
    no_reforest_infer = infer_lstm(observed_input,cfg)
    no_reforest_infer.to_csv('data/lstm_hybrid_no_reforest.csv')
    reforest_infer = infer_lstm(reforested_input,cfg)
    reforest_infer = reforest_infer[reforest_infer['year'] < 2016]
    no_reforest_infer = no_reforest_infer[no_reforest_infer['year'] < 2016]
    # reforest_infer.to_csv('data/lstm_hybrid_reforest.csv')
    # cesm_data = cesm_data[29::30].groupby(['year','lat','lon']).sum().reset_index()
    pd.set_option('display.max_columns', None)

    print(no_reforest_infer.describe())
    print(cesm_data.describe())
    print(reforest_infer.describe())
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
    observed_input = observed_input[observed_input['year'] < 2016]
    reforested_input = reforested_input[reforested_input['year'] < 2016]
    # plotComparison(cesm_data,reforest_infer,no_reforest_infer,'nppWood' )
    # plotComparison(cesm_data,reforest_infer,no_reforest_infer,'nppRoot')
    # plotComparison(cesm_data,reforest_infer,no_reforest_infer,'nppLeaf')
    cesm_data['agb'] = cesm_data['cStem'] + cesm_data['cLeaf'] + cesm_data['cOther']
    reforest_infer['agb'] = reforest_infer['cStem'] + reforest_infer['cLeaf'] + reforest_infer['cOther']
    no_reforest_infer['agb'] = no_reforest_infer['cStem'] + no_reforest_infer['cLeaf'] + no_reforest_infer['cOther']
    reforest_infer['area'] = reforest_infer.apply(lambda x: getArea(x['lat'],x['lon']),axis=1)
    reforest_infer['agb'] = reforest_infer['agb'] * reforest_infer['area'] / 1e9
    reforest_infer['cSoilAbove1m'] = reforest_infer['cSoilAbove1m'] * reforest_infer['area'] / 1e9

    no_reforest_infer['area'] = reforest_infer.apply(lambda x: getArea(x['lat'],x['lon']),axis=1)
    no_reforest_infer['agb'] = no_reforest_infer['agb'] * no_reforest_infer['area'] / 1e9
    no_reforest_infer['cSoilAbove1m'] = no_reforest_infer['cSoilAbove1m'] * no_reforest_infer['area'] / 1e9

    # plotComparison(cesm_data,reforest_infer,no_reforest_infer,'cLeaf')
    # plotComparison(cesm_data,reforest_infer,no_reforest_infer,'cStem')
    # plotComparison(cesm_data,reforest_infer,no_reforest_infer,'cOther')
    # plotComparison(cesm_data,reforest_infer,no_reforest_infer,'agb')

    plotTreeFracComparison(reforested_input,observed_input)
    # plotComparison(cesm_data,reforest_infer,no_reforest_infer,'cSoilAbove1m')
    
    plotDifference(reforest=reforest_infer,no_reforest=no_reforest_infer)
main()