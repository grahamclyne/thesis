#INFER MODEL W HYBRID DATA 
import pandas as pd
from hydra import initialize, compose


from pickle import load 
from lstm_model import RegressionLSTM
from transformer.transformer_model import CMIPTimeSeriesDataset
import numpy as np
import torch as T 
from preprocessing.utils import getArea
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import hydra
import numpy as np
from sklearn.preprocessing import StandardScaler
    




def infer_lstm(data,cfg):

    final_input = data[cfg.model.input + cfg.model.output + cfg.model.id]
    # scaler = load(open(f'{cfg.project}/checkpoint/lstm_scaler.pkl', 'rb'))
    # out_scaler = load(open(f'{cfg.project}/checkpoint/lstm_output_scaler.pkl', 'rb'))
    final_input.loc[:,cfg.model.input] = StandardScaler().fit_transform(final_input.loc[:,cfg.model.input])
    hold_out_out_scaler = StandardScaler().fit(final_input[29::30].loc[:,cfg.model.output])
    print(final_input.loc[:,cfg.model.output])
    model = RegressionLSTM(num_sensors=len(cfg.model.input), hidden_units=cfg.model.params.hidden_units,cfg=cfg)
    checkpoint = T.load(f'{cfg.project}/checkpoint/lstm_checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    final_input = data[cfg.model.input + cfg.model.id]
    ds = CMIPTimeSeriesDataset(final_input,cfg.model.params.seq_len,len(cfg.model.input) + len(cfg.model.id),cfg)
    batch_size = 1
    ldr = T.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=False)
    results = []
    ids = []
    model.eval()
    print(len(ds))
    for X,tgt,id in ldr:
        y = model(X.float())
        y = hold_out_out_scaler.inverse_transform(y.detach().numpy())
        results.extend(y)
        ids.append(id.detach().numpy())
        # print(id)
    ids = np.concatenate(ids)
    results_df = pd.DataFrame(results)
    results_df['year'] = ids[:,0]
    results_df['lat'] = ids[:,1]
    results_df['lon'] = ids[:,2]
    # results_df['treeFrac'] = scaler.invers_transform(final_input[29::30].groupby('year').sum()['treeFrac'])
    results_df.columns = cfg.model.output + ['year','lat','lon']
    print('results_df',results_df)   
    return results_df


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    cesm_data = pd.read_csv('data/timeseries_cesm_hold_out_data_30.csv')
    cesm_data.rename(columns={'# year':'year'},inplace=True)

    cesm_data = cesm_data[cfg.model.input + cfg.model.output + cfg.model.id]

    # cesm_data = cesm_data.iloc[0:300]

    # cesm_data = cesm_data[cesm_data['year'] == 2014]
    # reforested_input = pd.read_csv('data/observed_reforest_ts.csv')
    # cesm_data = cesm_data.where(cesm_data['year'] > 1980).dropna()
    # cesm_data = cesm_data.assign(lat=round(cesm_data['lat'],6))
    #get only columns with fin_year of 2014
    # x = cesm_data['year'][29::30].reset_index()
    # x['index'] = x['index'] - 29
    # x.set_index('index',inplace=True)
    # t = x.loc[x.index.repeat(30)].reset_index()
    # cesm_data['fin_year'] = t['year']
    # cesm_data = cesm_data[cesm_data['fin_year'] > 1984]
    # cesm_data = cesm_data.groupby(['year','lat','lon']).mean().reset_index()
    # cesm_data = cesm_data[cesm_data.lat != 42.879582]
    	# 68.324608	-136.25
    # observed_input = pd.read_csv(cfg.environment.path.observed_input)

    # # observed_input.fillna(observed_input.median(),inplace=True)
    # # observed_input = observed_input.where((observed_input['year'] > 1984) & (observed_input['year'] < 2015)).dropna()
    # observed_input = observed_input[['year','lat','lon','treeFrac']]

    # reforested_input = pd.read_csv(cfg.environment.path.reforested_input)
    # # reforested_input.rename(columns={'year_y':'year'},inplace=True)
    # reforested_input['lat'] = round(reforested_input['lat'],6)
    # observed_input['lat'] = round(observed_input['lat'],6)
    # reforested_input.fillna(reforested_input.median(),inplace=True)
    # reforested_input = reforested_input.where((reforested_input['year'] > 1984) & (reforested_input['year'] < 2015)).dropna()
    # reforested_input = reforested_input[['year','lat','lon','treeFrac']]
    # df_merged = pd.merge(cesm_data,observed_input,on=['year','lat','lon'],how='left')
    # reforested_merged = pd.merge(cesm_data.copy(),reforested_input,on=['year','lat','lon'],how='left')

    # df_merged = df_merged.drop(columns=['treeFrac_x'])
    # df_merged = df_merged.rename(columns={'treeFrac_y':'treeFrac'})
    # reforested_merged = reforested_merged.drop(columns=['treeFrac_x'])
    # reforested_merged = reforested_merged.rename(columns={'treeFrac_y':'treeFrac'})
    # df_merged.fillna(df_merged.median(),inplace=True)
    # reforested_merged.fillna(reforested_merged.median(),inplace=True)
    # pd.set_option('display.max_columns', None)
    # print(df_merged[29::30].groupby(['year','lat','lon']).sum().reset_index().describe())
    # print(cesm_data.describe())













    no_reforest_infer = infer_lstm(cesm_data,cfg)
    no_reforest_infer.to_csv('data/lstm_hybrid_no_reforest.csv')
    # reforest_infer = infer_lstm(reforested_merged,cfg)
    # reforest_infer.to_csv('data/lstm_hybrid_reforest.csv')
    cesm_data = cesm_data[29::30].groupby(['year','lat','lon']).sum().reset_index()



    cesm_data = cesm_data[~((cesm_data.lat == 68.324608) & (cesm_data.lon == -136.25))]




    print(cesm_data.describe())
    print(no_reforest_infer.describe())


    # print(len(reforest_infer))
    print(len(no_reforest_infer))
    print(len(cesm_data))
    #plot results
    # reforest_infer['agb'] = reforest_infer['cStem'] + reforest_infer['cLeaf'] + reforest_infer['cOther']
    cesm_data['agb'] = cesm_data['cStem'] + cesm_data['cLeaf'] + cesm_data['cOther']
    no_reforest_infer['agb'] = no_reforest_infer['cStem'] + no_reforest_infer['cLeaf'] + no_reforest_infer['cOther']

    # reforest_infer['area'] = reforest_infer.apply(lambda x: getArea(x['lat'],x['lon']),axis=1)
    cesm_data['area'] = cesm_data.apply(lambda x: getArea(x['lat'],x['lon']),axis=1)
    no_reforest_infer['area'] = no_reforest_infer.apply(lambda x: getArea(x['lat'],x['lon']),axis=1)

    # reforest_infer['agb'] = reforest_infer['agb'] * reforest_infer['area'] / 1e9 #to megatonnes
    cesm_data['agb'] = cesm_data['agb'] * cesm_data['area'] / 1e9 # kg to megatonnes
    no_reforest_infer['agb'] = no_reforest_infer['agb'] * no_reforest_infer['area'] / 1e9 # kg to megatonnes


    fig, ax = plt.subplots()
    ax.plot(cesm_data.groupby('year')['agb'].sum(),label='CESM')
    # ax.plot(reforest_infer.groupby('year')['agb'].sum(),label='Reforestation')
    ax.plot(no_reforest_infer.groupby('year')['agb'].sum(),label='No Reforestation')
    ax.legend()
    ax.title.set_text('Canada AGB')
    fig.savefig('agb_lstm.png')



    fig, ax = plt.subplots()
    ax.set_ylabel('Tree Cover %')
    ax.plot(cesm_data.groupby('year')['treeFrac'].mean(),label='CESM')
    # ax.plot(reforested_merged.groupby('year')['treeFrac'].mean(),label='Reforestation')
    # ax.plot(df_merged.groupby('year')['treeFrac'].mean(),label='No Reforestation')
    ax.title.set_text('Canada Tree Cover')
    ax.legend()
    fig.savefig('lstm_tree_frac.png')



    fig, ax = plt.subplots()
    ax.set_ylabel('soil 1m')
    ax.plot(cesm_data.groupby('year')['cSoilAbove1m'].sum(),label='CESM')
    # ax.plot(reforest_infer.groupby('year')['cSoilAbove1m'].sum(),label='Reforestation')
    ax.plot(no_reforest_infer.groupby('year')['cSoilAbove1m'].sum(),label='No Reforestation')
    ax.title.set_text('Canada Tree Cover')
    ax.legend()
    fig.savefig('lstm_soil.png')
main()