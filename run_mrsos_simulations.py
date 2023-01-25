import pandas as pd
from model import Net,CMIPDataset
from pickle import load
import torch as T
import hydra
from omegaconf import DictConfig
import numpy as np
import plotly.graph_objects as go

import matplotlib.pyplot as plt
def plot3d(data):
    comparison = data[['cVeg','lat','lon']]
    print(comparison)
    output = comparison.pivot(index='lat', columns='lon', values='cVeg')
    fig = go.Figure(data=[go.Surface(z=output.values)])
    fig.update_layout(title='Simulated Permutation', autosize=True)
    fig.update_layout(
            scene = dict(
                yaxis = dict(
                    tickmode = 'array',
                    tickvals = list(range(0,len(output.index),3)),
                    ticktext = [round(x,2) for x in output.index][::3],
                    title='latitude'
                    ),
                xaxis = dict(
                    tickmode = 'array',
                    tickvals = list(range(0,len(output.columns),5)),
                    ticktext = output.columns[::5],
                    title='longitude'
                    ),
                zaxis = dict(
                    tickvals = list(range(0,16,5)),
                    ticktext = [0,5,15],
                    title = 'kg/m^2'
                ),
                camera_eye= dict(x= 0.5, y= -1.7, z=1.),
                aspectratio = dict(x= 1, y= 1, z= 0.2)
            )
            # xaxis = dict(tickmode = 'array',ticktext = comparison['lat']),
            # yaxis = dict(tickmode = 'array',tickvals = list(range(0,len(comparison['lon']))),ticktext = comparison['lon'])
            )




def run_simulation(final_input,cfg):
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
    results_df.columns = cfg.model.output + ['year','lat','lon']
    return results_df

def plotmultiplebar(data):
    transposed = map(list,zip(*data))
    for i in transposed:
        print(i)
        plt.bar(np.arange(len(i)),i)
    plt.show()
@hydra.main(version_base=None, config_path="conf", config_name="ann_config")
def main(cfg: DictConfig):
    summed = []
    final_input = pd.read_csv(f'{cfg.path.data}/cleaned_observed_ann_input.csv',header=0)
    final_input = final_input.dropna(how='any').reset_index(drop=True)
    for i in range(-10,10,1):
        print(i)
        temp_mrsos = final_input['treeFrac'] + (i)
        temp_mrsos = temp_mrsos.clip(lower=0,upper=100)
        # print(temp_mrsos)
        temp_data = final_input.copy()
        temp_data['treeFrac'] = temp_mrsos
        simulated = run_simulation(temp_data,cfg)
        summed.append(simulated[['cStem','cLeaf','cOther']].sum().to_list())
        # plot3d(simulated)
    simulated.to_csv('test.csv')
    plotmultiplebar(summed)
    # print(final_input['mrsos'].describe())


main()
