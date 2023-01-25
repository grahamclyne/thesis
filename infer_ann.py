import pandas as pd
from model import Net,CMIPDataset
from pickle import load
import torch as T
import hydra
from omegaconf import DictConfig
import numpy as np

@hydra.main(version_base=None, config_path="conf", config_name="ann_config")
def main(cfg: DictConfig):

    final_input = pd.read_csv(f'{cfg.path.data}/cleaned_observed_ann_input.csv',header=0)
    final_input = final_input.dropna(how='any').reset_index(drop=True)
    print(final_input)
    final_input = final_input[cfg.model.input + cfg.model.id]
    scaler = load(open(f'{cfg.path.project}/ann_scaler.pkl', 'rb'))
    scaled_input = scaler.transform(final_input.loc[:,cfg.model.input])
    T.set_printoptions(sci_mode=False)
    model = Net(len(cfg.model.input),len(cfg.model.output))

    checkpoint = T.load(f'{cfg.path.project}/ann_checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    ds = CMIPDataset(scaled_input,len(cfg.model.input),len(cfg.model.output))
    batch_size = 128
    ldr = T.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=False)
    results = []
    model.eval()
    for X in ldr:
        y = model(T.tensor(X[0].float()))
        results.extend(y.detach().numpy())
        # ids.append(id.detach().numpy())
        
    results_df = pd.DataFrame(results)
    results_df['year'] = final_input['year']
    results_df['lat'] = final_input['lat']
    results_df['lon'] = final_input['lon']
    print(results_df)
    results_df.columns = cfg.model.output + ['year','lat','lon']
    print(results_df)   
    results_df.to_csv(f'{cfg.path.data}/forest_carbon_observed.csv')

main()
