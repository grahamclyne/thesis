import torch
import pandas as pd
import hydra
from omegaconf import DictConfig
from pickle import load
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import captum
from lstm_model import RegressionLSTM
from lstm_model import CMIPTimeSeriesDataset
import numpy as np
import shap
import geopandas as gpd
from preprocessing.utils import getGeometryBoxes

def plotNSTransect(lon,coordinates):
    df = gpd.GeoDataFrame([],geometry=getGeometryBoxes(coordinates[coordinates['lon'] == lon]))
    canada = gpd.read_file(f'data/shapefiles/lpr_000b16a_e/lpr_000b16a_e.shp')
    canada  = canada.to_crs('4326')
    f, axes = plt.subplots(figsize=(10, 10))
    canada.plot(ax=axes,alpha=0.2)
    df.plot(ax=axes)
    plt.title(f'Forest North-South Transect at Longitude {lon}',fontsize=15)
    plt.savefig(f'figures/NS_transect_{lon}.png',bbox_inches='tight')

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


def visualizeShapleyValues(cfg,model,final_input,lon):
    final_input = final_input[(final_input['lon'] == lon)]
    scaler = load(open(f'{cfg.project}/checkpoint/lstm_scaler_{cfg.run_name}.pkl', 'rb'))
    final_input.loc[:,tuple(cfg.model.input)] = scaler.transform(final_input[cfg.model.input])
    final_input= final_input[cfg.model.input + cfg.model.id]
    ds = CMIPTimeSeriesDataset(final_input,cfg.model.seq_len,len(cfg.model.input) + len(cfg.model.id),cfg)
    final_input = torch.tensor(ds.data[:,:,:6]).float()
    e = shap.DeepExplainer(model,final_input)
    shap_values = e.shap_values(final_input)
    sum = (shap_values[0] + shap_values[1] + shap_values[2] + shap_values[3])/4
    # shap.summary_plot(sum.reshape(-1,6), features=final_input.reshape(-1,6), feature_names=['ps','tsl','treeFrac','pr','tas_DJF','tas_JJA'])
    shap.summary_plot(sum.reshape(-1,6), features=final_input.reshape(-1,6), feature_names=['ps','tsl','treeFrac','pr','tas_DJF','tas_JJA'])

    plt.savefig(f'figures/shapley_values_{lon}.png')

#plots for 1984-2014
def visualizeAttribution(attr,ds,lon,cfg,means,stds,target_name):
    scaler = load(open(f'{cfg.project}/checkpoint/lstm_scaler_{cfg.run_name}.pkl', 'rb'))
    data = scaler.inverse_transform(ds[:,:6])
    def plot_attr(sign,cmap):
        f,ax = plt.subplots(nrows=6,ncols=1,figsize=(10,10))
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
        f.suptitle(f'Attribution for {target_name}',fontsize=15)
        f.subplots_adjust(top=0.85)
        f.tight_layout(rect=[0, 0.03, 1, 0.95])
        f.savefig(f'figures/{sign}_attr_{lon}_{target_name}.png')
    plot_attr('positive','Greens')
    plot_attr('negative','Reds')

#get attribution for a single transect
def NS_transect_attr_data(df,cfg,lon,ig,target):
    transect = df[(df['lon'] == lon)]
    #reduce rolling window data back to single value
    transect = transect.groupby(['year','lat','lon']).mean().reset_index()
    total_attr = np.zeros((30,6))
    total_ds = np.zeros((30,9))
    for lat in transect.lat.unique():
        transect_lat = transect[(transect['year'] < 2015) & (transect['year'] > 1984) & (transect['lat'] == lat)]
        attr,ds = getAttribution(transect_lat[cfg.model.input + cfg.model.id],cfg,ig,target)
        total_attr = (total_attr + attr) / 2
        total_ds = (total_ds + ds.data[0,:,:]) / 2
    return total_attr,total_ds



@hydra.main(version_base=None, config_path="../conf", config_name="config")
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
    final_input = pd.read_csv(f'{cfg.data}/observed_timeseries{cfg.model.seq_len}_data.csv')
    ecozones_coordinates = pd.read_csv(f'{cfg.data}/ecozones_coordinates.csv')
    ecozones_coordinates = ecozones_coordinates[ecozones_coordinates['zone'].isin(['Boreal Cordillera','Boreal PLain', 'Boreal Shield'])]
    ecozones_coordinates['lat'] = ecozones_coordinates['lat'].round(6)
    final_input['lat'] = final_input['lat'].round(6)
    final_input = pd.merge(final_input,ecozones_coordinates,how='inner', on=['lat','lon'])
    means,stds = get_df_means_stds(final_input,cfg)
    target = 2 
    target_name = 'cStem'
    lon = -118.75
    visualizeShapleyValues(cfg,model,final_input,lon)

    lat = 56.073299
    attr,ds = NS_transect_attr_data(final_input,cfg,lon,ig,target)
    visualizeAttribution(attr,ds,lon,cfg,means,stds,target_name)
    plotNSTransect(lon,ecozones_coordinates)

    lon = -73.75
    lat = 49.476440
    attr,ds = NS_transect_attr_data(final_input,cfg,lon,ig,target)
    visualizeAttribution(attr,ds,lon,cfg,means,stds,target_name)
    plotNSTransect(lon,ecozones_coordinates)



main()

