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
from preprocessing.utils import scaleVariable
from sklearn.preprocessing import StandardScaler


def plotComparison(df1:pd.DataFrame,df2:pd.DataFrame,df3:pd.DataFrame,variable:str):
    fig, ax = plt.subplots()
    ax.plot(df1.groupby('year').sum()[variable],label='CESM')
    ax.plot(df2.groupby('year').sum()[variable],label='Reforestation')
    ax.plot(df3.groupby('year').sum()[variable],label='No Reforestation')
    ax.legend()
    ax.title.set_text(f'{variable}')
    # ax.axhline(0, color='black', linewidth=.5)

    fig.savefig(f'{variable}_lstm.png')


def infer_lstm(data,cfg):

    final_input = data[cfg.model.input + cfg.model.output + cfg.model.id]
    scaler = load(open(f'{cfg.project}/checkpoint/lstm_scaler.pkl', 'rb'))
    # out_scaler = load(open(f'{cfg.project}/checkpoint/lstm_output_scaler.pkl', 'rb'))
    # scaler = StandardScaler()
    print(final_input)
    final_input.loc[:,cfg.model.input] =scaler.transform(final_input.loc[:,cfg.model.input])

    # final_input[cfg.model.input]=final_input[cfg.model.input].apply(scaler.transform)
    print(final_input)
    # hold_out_out_scaler = StandardScaler().fit(final_input[29::30].loc[:,cfg.model.output])
    model = RegressionLSTM(num_sensors=len(cfg.model.input), hidden_units=cfg.model.params.hidden_units,cfg=cfg)
    checkpoint = T.load(f'{cfg.project}/checkpoint/lstm_checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    final_input = final_input[cfg.model.input + cfg.model.output + cfg.model.id]
    ds = CMIPTimeSeriesDataset(final_input,cfg.model.params.seq_len,len(cfg.model.input) + len(cfg.model.output) + len(cfg.model.id),cfg)
    batch_size = 1
    ldr = T.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=False)
    results = []
    ids = []
    tgts = []
    model.eval()
    for X,tgt,id in ldr:
        y = model(X.float())
        y = (y.detach().numpy())
        results.extend(y)
        ids.append(id.detach().numpy())
        tgts.extend(tgt.detach().numpy())
        print(y)
        print(tgt)
        # print(id)
    ids = np.concatenate(ids)
    results_df = pd.DataFrame(results)
    tgts_df = pd.DataFrame(tgts)
    tgts_df['year'] = ids[:,0]
    tgts_df['lat'] = ids[:,1]
    tgts_df['lon'] = ids[:,2]
    results_df['year'] = ids[:,0]
    results_df['lat'] = ids[:,1]
    results_df['lon'] = ids[:,2]
    tgts_df.columns = cfg.model.output + ['year','lat','lon']
    results_df.columns = cfg.model.output + ['year','lat','lon']
    # print(results_df)   
    return results_df,tgts_df


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cesm_data = pd.read_csv('data/timeseries_cesm_hold_out_data_30.csv')
    cesm_data.rename(columns={'# year':'year'},inplace=True)

    cesm_data = cesm_data[cfg.model.input + cfg.model.output + cfg.model.id]
    # cesm_data = cesm_data.where(cesm_data['year'] > 1984).dropna()
    # cesm_data = cesm_data.iloc[0:300]

    # cesm_data = cesm_data[cesm_data['year'] == 2014]
    # reforested_input = pd.read_csv('data/observed_reforest_ts.csv')
    # cesm_data = cesm_data.where(cesm_data['year'] > 1984)
    cesm_data = cesm_data.assign(lat=round(cesm_data['lat'],6))
    #get only columns with fin_year of 2014
    # x = cesm_data['year'][29::30].reset_index()
    # x['index'] = x['index'] - 29
    # x.set_index('index',inplace=True)
    # t = x.loc[x.index.repeat(30)].reset_index()
    # cesm_data['fin_year'] = t['year']
    # cesm_data = cesm_data[cesm_data['fin_year'] > 1984]
    # cesm_data = cesm_data.groupby(['year','lat','lon']).mean().reset_index()
    cesm_data = cesm_data[cesm_data.lat != 42.879582]
    # cesm_data.dropna(how='any',inplace=True)
    	# 68.324608	-136.25
    observed_input = pd.read_csv(cfg.environment.path.observed_input)

    # observed_input.fillna(observed_input.median(),inplace=True)
    # observed_input = observed_input.where((observed_input['year'] > 1980) & (observed_input['year'] < 2015)).dropna()
    observed_input = observed_input[['year','lat','lon','treeFrac']]
    reforested_input = pd.read_csv(cfg.environment.path.reforested_input)
    # reforested_input['lat'] = round(reforested_input['lat'],6)
    # observed_input['lat'] = round(observed_input['lat'],6)
    # reforested_input.fillna(reforested_input.median(),inplace=True)
    # reforested_input = reforested_input.where((reforested_input['year'] > 1984) & (reforested_input['year'] < 2015)).dropna()
    reforested_input = reforested_input[['year','lat','lon','treeFrac']]
    # print(observed_input.where(observed_input['treeFrac']))
    df_merged = pd.merge(cesm_data,observed_input,on=['year','lat','lon'],how='inner')
    reforested_merged = pd.merge(cesm_data,reforested_input,on=['year','lat','lon'],how='inner')
    df_merged = df_merged.drop(columns=['treeFrac_x'])
    df_merged = df_merged.rename(columns={'treeFrac_y':'treeFrac'})
    # pd.set_option('display.max_rows', None)
    # empty_coords = df_merged[df_merged['treeFrac'].isna()].dropna(how='all')[['lat','lon']].drop_duplicates().reset_index()
    # print(empty_coords)
    # # print(df_merged.where(df_merged['lat'].isin(empty_coords['lat'])))
    # df_merged = df_merged.where(~((df_merged['lat'].isin(empty_coords['lat'])) & (df_merged['lon'].isin(empty_coords['lon']))))






    reforested_merged = reforested_merged.drop(columns=['treeFrac_x'])
    reforested_merged = reforested_merged.rename(columns={'treeFrac_y':'treeFrac'})
    df_merged.interpolate(method='spline',order=5,inplace=True)
    reforested_merged.interpolate(method='spline',order=5,inplace=True)
    reforested_merged.dropna(how='any',inplace=True)
    no_reforest_infer,tgts = infer_lstm(cesm_data,cfg)
    no_reforest_infer.to_csv('data/lstm_hybrid_no_reforest.csv')
    reforest_infer,tgts1 = infer_lstm(reforested_merged,cfg)
    
    # reforest_infer.to_csv('data/lstm_hybrid_reforest.csv')
    cesm_data = cesm_data[29::30].groupby(['year','lat','lon']).sum().reset_index()
    pd.set_option('display.max_columns', None)

    print(no_reforest_infer.describe())
    print(tgts.describe())
    print(cesm_data.describe())
    
    # cesm_data = cesm_data[~((cesm_data.lat == 68.324608) & (cesm_data.lon == -136.25))]
    # print(cesm_data.describe())
    # print(no_reforest_infer.describe())
    for var in cfg.model.output:
        if(var == "cSoilAbove1m"):
            continue
        reforest_infer.loc[:,var] = reforest_infer.loc[:,var] / 1000000000
        no_reforest_infer.loc[:,var] = no_reforest_infer.loc[:,var] / 1000000000


    plotComparison(cesm_data,reforest_infer,no_reforest_infer,'nppWood')
    plotComparison(cesm_data,reforest_infer,no_reforest_infer,'nppRoot')
    plotComparison(cesm_data,reforest_infer,no_reforest_infer,'nppLeaf')
    plotComparison(cesm_data,reforested_merged,df_merged,'treeFrac')
    plotComparison(cesm_data,reforest_infer,no_reforest_infer,'cSoilAbove1m')

main()