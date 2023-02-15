import pandas as pd

from pickle import load
import torch as T
from transformer.transformer_model import CMIPTimeSeriesDataset
from lstm_model import RegressionLSTM
import hydra
from omegaconf import DictConfig
import numpy as np

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    final_input = pd.read_csv(f'{cfg.data}/observed_timeseries30_data.csv')
    print(final_input)

    final_input = final_input.dropna(how='any')
    final_input = final_input[cfg.model.input + cfg.model.id]
    scaler = load(open(f'{cfg.project}/checkpoint/lstm_scaler.pkl', 'rb'))
    out_scaler = load(open(f'{cfg.project}/checkpoint/lstm_output_scaler.pkl', 'rb'))
    final_input.loc[:,cfg.model.input] = scaler.transform(final_input.loc[:,cfg.model.input])
    # final_input.loc[:,cfg.model.output] = out_scaler.transform(final_input.loc[:,cfg.model.output])

    model = RegressionLSTM(num_sensors=len(cfg.model.input), hidden_units=cfg.model.params.hidden_units,cfg=cfg)
    checkpoint = T.load(f'{cfg.project}/checkpoint/lstm_checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ds = CMIPTimeSeriesDataset(final_input,cfg.model.params.seq_len,len(cfg.model.input) + len(cfg.model.id),cfg)
    batch_size = 1
    ldr = T.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=False)
    results = []
    ids = []
    model.eval()
    print(len(ldr))
    for X,tgt,id in ldr:
        print(X)
        y = model(T.tensor(X.float()))
        # print(X[0])
        y = out_scaler.inverse_transform(y.detach().numpy())
        results.extend(y)
        print(tgt[0])
        print(id[0])
        ids.append(id.detach().numpy())
        
    ids = np.concatenate(ids)
    results_df = pd.DataFrame(results)
    results_df['year'] = ids[:,0]
    results_df['lat'] = ids[:,1]
    results_df['lon'] = ids[:,2]
    # output = results_df.groupby('year').mean()
    results_df.columns = cfg.model.output + ['year','lat','lon']
    print(results_df)   
    results_df.to_csv(f'{cfg.data}/forest_carbon_observed_lstm.csv')

main()