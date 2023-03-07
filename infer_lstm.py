import pandas as pd

from pickle import load
import torch as T
from transformer.transformer_model import CMIPTimeSeriesDataset
from lstm_model import RegressionLSTM
import hydra
from omegaconf import DictConfig
import numpy as np


def infer_lstm(data,cfg):
    model_name = cfg.run_name
    final_input = data[cfg.model.input +cfg.model.id]
    scaler = load(open(f'{cfg.project}/checkpoint/lstm_scaler_{model_name}.pkl', 'rb'))
    # out_scaler = load(open(f'{cfg.project}/checkpoint/lstm_output_scaler.pkl', 'rb'))
    # scaler = StandardScaler()
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
def main(cfg: DictConfig,):
    final_input = pd.read_csv(f'{cfg.data}/observed_timeseries30_data.csv')


    # final_input = pd.read_csv(f'{cfg.data}/timeseries_cesm_training_data_30.csv')
    # final_input.rename(columns={'# year':'year'},inplace=True)



    run_name = cfg.run_name
    final_input = final_input.dropna(how='any')
    final_input = final_input[cfg.model.input + cfg.model.id]
    scaler = load(open(f'{cfg.project}/checkpoint/lstm_scaler_{run_name}.pkl', 'rb'))
    # out_scaler = load(open(f'{cfg.project}/checkpoint/lstm_output_scaler_soil_only.pkl', 'rb'))
    # final_input['tsl'] = final_input['tsl'].apply(lambda x: x-4)
    final_input.loc[:,cfg.model.input] = scaler.transform(final_input.loc[:,cfg.model.input])
    # final_input.loc[:,cfg.model.output] = out_scaler.transform(final_input.loc[:,cfg.model.output])

    model = RegressionLSTM(num_sensors=len(cfg.model.input), hidden_units=cfg.model.hidden_units,cfg=cfg)
    checkpoint = T.load(f'{cfg.project}/checkpoint/lstm_checkpoint_{run_name}.pt')

    #in case distributed training was used
    for key in list(checkpoint['model_state_dict'].keys()):
        checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'].pop(key)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ds = CMIPTimeSeriesDataset(final_input,cfg.model.seq_len,len(cfg.model.input) + len(cfg.model.id),cfg)
    batch_size = 1
    ldr = T.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=False)
    results = []
    ids = []
    model.eval()
    for X,tgt,id in ldr:
        # print(X)
        y = model(T.tensor(X.float()))
        # print(X[0])
        y = (y.detach().numpy())
        results.extend(y)
        # print(tgt[0])
        # print(id[0])
        ids.append(id.detach().numpy())
        
    ids = np.concatenate(ids)
    results_df = pd.DataFrame(results)
    results_df['year'] = ids[:,0]
    results_df['lat'] = ids[:,1]
    results_df['lon'] = ids[:,2]
    # output = results_df.groupby('year').mean()
    results_df.columns = cfg.model.output + ['year','lat','lon']

    # results_df = infer_lstm(final_input,cfg)
    print(results_df)   
    print(results_df.groupby('year').sum())
    results_df.to_csv(f'{cfg.data}/forest_carbon_observed_lstm.csv')



main(  )