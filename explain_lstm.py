import torch
import pandas as pd
import hydra
from omegaconf import DictConfig
from pickle import load
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import captum
from lstm_model import RegressionLSTM
from transformer.transformer_model import CMIPTimeSeriesDataset
import numpy as np
from visualization.plot_helpers import plotNSTransect


#gets mean,std for 1984-2014
def get_df_means_stds(df,cfg):
    df = df.groupby(['year','lat','lon']).mean().reset_index()[df['year'] < 2015]
    means = []
    stds = []
    for var in cfg.model.input:
        means.append(df[var].mean())
        stds.append(df[var].std())
    return means,stds

def getAttribution(input,cfg,ig,target):
    scaler = load(open(f'{cfg.project}/checkpoint/lstm_scaler_{cfg.run_name}.pkl', 'rb'))
    input.loc[:,tuple(cfg.model.input)] = scaler.transform(input[cfg.model.input])
    input= input[cfg.model.input + cfg.model.id]
    ds = CMIPTimeSeriesDataset(input,cfg.model.seq_len,len(cfg.model.input) + len(cfg.model.id),cfg)
    input = torch.tensor(ds.data[:,:,:6]).float()
    attr= ig.attribute(input,target=target) #target 2 is cStem
    return attr.detach().numpy(),ds


#plots for 1984-2014
def visualizeAttribution(attr,ds,lon,cfg,means,stds,target_name):
    scaler = load(open(f'{cfg.project}/checkpoint/lstm_scaler_{cfg.run_name}.pkl', 'rb'))
    data = scaler.inverse_transform(ds[:,:6])

    def plot_attr(sign,cmap):
        f,ax = plt.subplots(nrows=6,ncols=1,figsize=(10,10))
        # plt.title(f'North-South Transect (Longitude={lon}, {sign.capitalize()} Attribution for {target_name}',fontsize=20)
        for single_ax in range(len(ax)):
            ax[single_ax].set_xlim([0, 29])
            #set y limits to be 2 standard deviations above and below the mean
            ax[single_ax].set_ylim([means[single_ax]-(stds[single_ax]*2), means[single_ax]+(stds[single_ax]*2)])
            ax[single_ax].tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        ax[single_ax].tick_params(axis='x',which='both',bottom=True,top=False,labelbottom=True)
        plt.xticks(np.arange(0,30, step=1),labels=np.arange(1985, 2015, step=1),rotation=45)
        plt.xlabel('Year')
        f,ax = captum.attr.visualization.visualize_timeseries_attr(attr[0,:,:],data,plt_fig_axis=(f,ax),
                                                        method="overlay_individual",
                                                        channel_labels=['ps','tsl','treeFrac','pr','tas_DJF','tas_JJA'],
                                                        cmap=cmap,
                                                        sign=sign, 
                                                        use_pyplot=False)
        f.suptitle(f'North-South Transect (Longitude={lon}, {sign.capitalize()} Attribution for {target_name}',fontsize=15)
        f.subplots_adjust(top=0.85)
        f.tight_layout(rect=[0, 0.03, 1, 0.95])

        f.savefig(f'{sign}_attr_{lon}_NS_transect_{target_name}.png')
    plot_attr('positive','Greens')
    plot_attr('negative','Reds')
    plotNSTransect(lon)

#get attribution for a single transect
def NS_transect_attr_data(df,cfg,lon,ig,target):
    transect = df[df['lon'] == lon]

    #reduce rolling window data back to single value
    transect = transect.groupby(['year','lat','lon']).mean().reset_index()
    total_attr = np.zeros((30,6))
    total_ds = np.zeros((30,9))
    for lat in transect.lat.unique():
        transect_lat = transect[(transect['lat'] == lat) & (transect['lon'] == lon)]
        transect_lat = transect_lat[(transect_lat['year'] < 2015) & (transect_lat['year'] > 1984)]
        attr,ds = getAttribution(transect_lat[cfg.model.input + cfg.model.id],cfg,ig,target)
        total_attr = (total_attr + attr) / 2
        total_ds = (total_ds + ds.data[0,:,:]) / 2
    return total_attr,total_ds

    
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    model = RegressionLSTM(num_sensors=len(cfg.model.input), hidden_units=cfg.model.hidden_units,cfg=cfg)
    model_name = cfg.run_name
    checkpoint = torch.load(f'{cfg.project}/checkpoint/lstm_checkpoint_{model_name}.pt')
    #if distributed training, remove module. from keys
    for key in list(checkpoint['model_state_dict'].keys()):
        checkpoint['model_state_dict'][key.replace('module.', '')] = checkpoint['model_state_dict'].pop(key)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    ig = IntegratedGradients(model)
    final_input = pd.read_csv(f'{cfg.data}/observed_timeseries30_data.csv')
    means,stds = get_df_means_stds(final_input,cfg)
    target = 3 # cSoilAbove1m
    target_name = 'cSoilAbove1m'
    # target = 2 # cStem
    # target_name = 'cStem'
    lon = -73.75
    attr,ds = NS_transect_attr_data(final_input,cfg,lon,ig,target)
    visualizeAttribution(attr,ds,lon,cfg,means,stds,target_name)
    lon = -118.75
    attr,ds = NS_transect_attr_data(final_input,cfg,lon,ig,target)
    visualizeAttribution(attr,ds,lon,cfg,means,stds,target_name)
main()