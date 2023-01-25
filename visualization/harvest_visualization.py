import pandas as pd
from other.utils import getArea,getCoordinates,readCoordinates
from functools import reduce
from plotly import graph_objects as go
import other.config as config
#to run: cwd to thesis, python -m visualization.harvest_visualization
import hydra
from omegaconf import DictConfig

def getYearlyDeforestation(cfg):
    

@hydra.main(version_base=None, config_path="../conf", config_name="ann_config")
def main(cfg: DictConfig):

    harvest_df = pd.read_csv(f'{config.GENERATED_DATA}/nfis_harvest_data.csv',header=None)
    #the year columns are numbered 0,31
    harvest_df.columns = [x for x in range(0,33)] + ['year','lat','lon']
    ordered_latitudes = readCoordinates(f'{cfg.path.data}/grid_latitudes.csv',is_grid_file=True)
    ordered_longitudes = readCoordinates(f'{cfg.path.data}/grid_longitudes.csv',is_grid_file=True)
    total_harvested = 0

    #getting actual total_pixel size and "no_change" pixels, harvest data has 0's as no harvest which blend in with the cropped zeroes
    nfis_df = pd.read_csv(f'{config.GENERATED_DATA}/nfis_tree_cover_data.csv')

    #only need one year, all we are getting is the pixel size of each grid cell 
    nfis_df = nfis_df[nfis_df['year'] == 1985]
    data_frames = [harvest_df,nfis_df]
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['year','lat','lon'],
                                                how='outer'), data_frames)

    #scale each number of harvest pixels by the total num of pixels
    for i in range(1,31):
        df_merged[i] = df_merged[i] / (df_merged['total_pixels'] - df_merged['no_change']).fillna(0)

    areas = []
    for i,row in df_merged.iterrows():
        area = getArea(*getCoordinates((row['lat'],row['lon']),ordered_latitudes,ordered_longitudes))
        areas.append(area)

    areas_df = pd.DataFrame(areas)

    harvested_totals_per_year = []
    for index in range(1,31):
        harvested_totals_per_year.append((df_merged[index] * areas_df[0]).sum()/10000)#division by 10000 for 


    fig = go.Figure()
    years = [x for x in range(1984,2019,1)]
    fig.add_trace(go.Scatter(x=years, y=harvested_totals_per_year,
                        mode='markers',
                        name='nfis'
                    ))
    fig.update_layout(title='harvested fraction')
    fig.update_xaxes(title_text="year")
    fig.update_yaxes(title_text="hectares",zeroline=True,rangemode='tozero')

    fig.show()
    # nfis_tree_cover = pd.read_csv('/Users/gclyne/thesis/data/generated_observable_data/nfis_tree_cover_data.csv')
    # nfis_tree_cover['observed_tree_cover'] = nfis_tree_cover['coniferous'] + nfis_tree_cover['broadleaf'] + nfis_tree_cover['mixedwood'] + nfis_tree_cover['wetland-treed']

    # areas = []
    # for i,row in nfis_tree_cover.iterrows():
    #     area = getArea(*getCoordinates((row['lat'],row['lon']),ordered_latitudes,ordered_longitudes))
    #     areas.append(area)

    # nfis_tree_cover['observed_tree_cover'] = (nfis_tree_cover['observed_tree_cover'] / (nfis_tree_cover['total_pixels'] - nfis_tree_cover['no_change'])).fillna(0)
    # nfis_tree_cover['observed_tree_cover'] = nfis_tree_cover['observed_tree_cover'] * areas
    # nfis_grouped = nfis_tree_cover.groupby('year').sum()
    # harvested_totals_per_year = [x/10000 for x in harvested_totals_per_year]

    # fig = go.Figure()
    # years = [x for x in range(1984,2019,1)]
    # fig.add_trace(go.Scatter(x=years, y=nfis_grouped['observed_tree_cover'] / 10000,
    #                     mode='markers',
    #                     name='nfis'
    #                 ))
    # # fig.add_trace(go.Scatter(x=years, y=harvested_totals_per_year[:-2],
    # #                     mode='lines',
    # #                     name='nfis',
    # # ))
    # fig.update_layout(title='tree cover fraction')
    # fig.update_xaxes(title_text="year")
    # fig.update_yaxes(title_text="hectares")

    # fig.show()

main()