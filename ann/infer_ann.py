import pandas as pd
from ann.ann_model import Net,CMIPDataset
from pickle import load
import torch as T
import hydra
from omegaconf import DictConfig
import numpy as np


def generate_ann_predictions(cfg: DictConfig,input,output):
    final_input = pd.read_csv(input,header=0)
    # final_input = final_input.dropna(how='any').reset_index(drop=True)

    #replace 0 with column median 
    for col in final_input.columns:
        if col not in ['year','lat','lon']:
            # data = final_input[col]
            median = final_input[col].median()
            # data=data.replace(0,median)
            final_input.loc[:,col]=final_input.loc[:,col].replace(0,median)

    final_input = final_input.fillna(final_input.median())

    final_input = final_input[cfg.model.input + cfg.model.id]
    scaler = load(open(f'{cfg.path.project}/checkpoint/ann_scaler.pkl', 'rb'))
    scaled_input = scaler.transform(final_input.loc[:,cfg.model.input])
    model = Net(len(cfg.model.input),len(cfg.model.output))
    checkpoint = T.load(f'{cfg.path.project}/checkpoint/ann_checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    ds = CMIPDataset(scaled_input,len(cfg.model.input),len(cfg.model.output))
    batch_size = 128
    ldr = T.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=False)
    results = []
    model.eval()
    for X in ldr:
        y = model(X[0].float())
        results.extend(y.detach().numpy())        
    results_df = pd.DataFrame(results)
    results_df['year'] = final_input['year']
    results_df['lat'] = final_input['lat']
    results_df['lon'] = final_input['lon']
    results_df.columns = cfg.model.output + ['year','lat','lon']
    results_df.to_csv(output,index=False)


@hydra.main(version_base=None, config_path="../conf", config_name="ann_config")
def main(cfg: DictConfig):
    generate_ann_predictions(cfg,cfg.path.observed_input,cfg.path.observed_estimates)
    generate_ann_predictions(cfg,cfg.path.reforested_input,cfg.path.reforested_estimates)


main()
