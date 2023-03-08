import pandas as pd

from pickle import load
import torch as T
from transformer.transformer_model import CMIPTimeSeriesDataset
from lstm_model import RegressionLSTM
# import hydra
# from omegaconf import DictConfig
import numpy as np


def infer_lstm(cfg,tsl_offset=0):
    final_input = pd.read_csv(f'{cfg.data}/observed_timeseries30_data.csv')
    model_name = cfg.run_name
    final_input = final_input[cfg.model.input +cfg.model.id]
    scaler = load(open(f'{cfg.project}/checkpoint/lstm_scaler_{model_name}.pkl', 'rb'))
    # final_input = final_input.where(final_input['tsl'] > 0)
    final_input['tas_DJF'] = final_input['tas_DJF'].apply(lambda x: x+tsl_offset)
    final_input['tas_JJA'] = final_input['tas_JJA'].apply(lambda x: x+tsl_offset)
    final_input['tsl'] = final_input['tsl'].apply(lambda x: x+tsl_offset)

    final_input.loc[:,cfg.model.input ] =scaler.transform(final_input.loc[:,cfg.model.input])
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
    model.eval()
    for X,_,id in ldr:
        y = model(X.float())
        y = (y.detach().numpy())
        results.extend(y)
        ids.append(id.detach().numpy())
    ids = np.concatenate(ids)
    results_df = pd.DataFrame(results)
    results_df['year'] = ids[:,0]
    results_df['lat'] = ids[:,1]
    results_df['lon'] = ids[:,2]
    results_df.columns = cfg.model.output + ['year','lat','lon']
    return results_df


# @hydra.main(version_base=None, config_path="conf", config_name="config")
# def main(cfg: DictConfig):
#     infer_lstm(

# main(  )