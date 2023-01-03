import pandas as pd

from pickle import load
import torch as T
from transformer_no_decoder import TimeSeriesTransformer,CMIPTimeSeriesDataset
import hydra
from omegaconf import DictConfig
import numpy as np

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    final_input = pd.read_csv(f'{cfg.path.data}/observed_transformer_data.csv')
    final_input = final_input.dropna(how='any')
    final_input = final_input[cfg.model.input + cfg.model.id]
    scaler = load(open(f'{cfg.path.data}/trans_train_scaler.pkl', 'rb'))
    # data_to_estimate = scaler.transform(final_input[cfg.model.input])
    final_input.loc[:,cfg.model.input] = scaler.transform(final_input.loc[:,cfg.model.input])
    T.set_printoptions(sci_mode=False)
    print(final_input[cfg.model.input])
    model = TimeSeriesTransformer(
        input_feature_size=len(cfg.model.input),
        input_seq_len=cfg.trans_params.input_seq_len,
        batch_first=cfg.trans_params.batch_first,
        dim_val=cfg.trans_params.dim_val,
        n_encoder_layers=cfg.trans_params.num_encoder_layers,
        n_heads=cfg.trans_params.num_heads,
        num_predicted_features=len(cfg.model.output),
        dropout_encoder=cfg.trans_params.dropout_encoder,
        dropout_pos_enc=cfg.trans_params.dropout_pos_enc,
        dim_feedforward_encoder=cfg.trans_params.dim_feedforward_encoder
                           )
    checkpoint = T.load(f'{cfg.path.data}/mdl_checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    ds = CMIPTimeSeriesDataset(final_input,cfg.trans_params.input_seq_len,len(cfg.model.input) + len(cfg.model.id),cfg)
    batch_size = 128
    ldr = T.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=False)
    results = []
    ids = []
    model.eval()
    for X,tgt,id in ldr:
        y = model(T.tensor(X.float()))
        # print(X[0])
        results.extend(y.detach().numpy())
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
    results_df.to_csv(f'{cfg.path.data}/forest_carbon_observed_transformer.csv')

main()