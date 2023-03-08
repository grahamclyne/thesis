import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import box
import geopandas
import hydra
from omegaconf import DictConfig
import pickle
from preprocessing.utils import getArea
def get_box(input):
    bc_coords = list(zip(input.lat,input.lon))
    re_bc_boxes = []
    ordered_lats = pd.read_csv('data/grid_latitudes.csv',header=None)
    ordered_lons = pd.read_csv('data/grid_longitudes.csv',header=None)
    for bottom,left in bc_coords:
        bottom = round(bottom,6)
        lat_index = ordered_lats.loc[round(ordered_lats[0],6) == bottom].index[0] + 1
        top = ordered_lats.iloc[lat_index][0]
        lon_index = ordered_lons.loc[ordered_lons[0] == left].index[0] + 1
        right = ordered_lons.iloc[lon_index][0]
        bbox = box(left,bottom,right,top)
        re_bc_boxes.append(bbox)
    return re_bc_boxes


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    #read in data
    no_reforestation = pd.read_csv(f'{cfg.data}/hybrid_no_reforest.csv')
    total_reforestation = pd.read_csv(f'{cfg.data}/hybrid_reforest.csv')
    canada = geopandas.read_file(f'{cfg.data}/shapefiles/lpr_000b16a_e/lpr_000b16a_e.shp')
    with open(f'{cfg.data}/provincial_coords.pickle', 'rb') as handle:
        provincial_dict = pickle.load(handle) 

    
    no_reforestation = no_reforestation[(no_reforestation['year'] > 1984) & (no_reforestation['year'] < 2015)]
    total_reforestation['agb'] = total_reforestation['cStem'] + total_reforestation['cLeaf'] + total_reforestation['cOther']
    no_reforestation['agb'] = no_reforestation['cStem'] + no_reforestation['cLeaf'] + no_reforestation['cOther']
    total_reforestation['area'] = total_reforestation.apply(lambda x: getArea(x['lat'],x['lon']),axis=1)
    no_reforestation['area'] = no_reforestation.apply(lambda x: getArea(x['lat'],x['lon']),axis=1)
    total_reforestation['agb'] = total_reforestation['agb'] * total_reforestation['area'] / 1e9 #to megatonnes
    no_reforestation['agb'] = no_reforestation['agb'] * no_reforestation['area'] / 1e9 # kg to megatonnes

    bc_lats = pd.DataFrame([x[0] for x in provincial_dict['British Columbia / Colombie-Britannique']])
    bc_lons = pd.DataFrame([x[1] for x in provincial_dict['British Columbia / Colombie-Britannique']])
    bc = pd.concat([bc_lats,bc_lons],axis=1)
    bc.columns = ['lon','lat']
    bc.lat = round(bc.lat,6)
    re_bc = pd.merge(total_reforestation,bc,how='inner',left_on=['lat','lon'],right_on=['lat','lon'])
    no_bc = pd.merge(no_reforestation,bc,how='inner',left_on=['lat','lon'],right_on=['lat','lon'])


    re_bc_2015 = re_bc[re_bc['year'] ==2015].reset_index(drop=True)
    no_bc_2015 = no_bc[no_bc['year'] ==2015].reset_index(drop=True)
    observed_input = pd.read_csv(f'{cfg.data}/cleaned_observed_ann_input.csv')
    reforested_input =  pd.read_csv(f'{cfg.data}/total_reforestation.csv')
    observed_input = pd.merge(observed_input,bc,how='inner',left_on=['lat','lon'],right_on=['lat','lon'])
    reforested_input = pd.merge(reforested_input,bc,how='inner',left_on=['lat','lon'],right_on=['lat','lon'])

    observed_input = observed_input[observed_input['year'] == 2014].reset_index(drop=True)
    reforested_input = reforested_input[reforested_input['year'] == 2014].reset_index(drop=True)

    boxes = get_box(no_bc_2015)
    tree_boxes = get_box(observed_input)
    canada_boxes = get_box(no_reforestation[no_reforestation['year'] == 2014])
    canada = canada.to_crs('4326')
    gdf_diff = geopandas.GeoDataFrame(
        (re_bc_2015['agb'] - no_bc_2015['agb']), geometry=boxes)
    treeFrac_diff = geopandas.GeoDataFrame(
        (reforested_input['treeFrac'] - observed_input['treeFrac']), geometry=tree_boxes)
    canada_carbon = geopandas.GeoDataFrame(no_reforestation[no_reforestation['year'] == 2014]['agb'],geometry=canada_boxes)
    canada_carbon_2013 = geopandas.GeoDataFrame(no_reforestation[no_reforestation['year'] == 2013]['agb'],geometry=canada_boxes)


    #plot carbon gained from reforestation
    fig, ax = plt.subplots()
    cbar=plt.cm.ScalarMappable(cmap='Greens',norm=plt.Normalize(vmin=gdf_diff['agb'].min(), vmax=gdf_diff['agb'].max()))
    canada[canada.PRUID=='59'].plot(ax=ax, color='white', edgecolor='black')
    gdf_diff.plot(ax=ax,column='agb',cmap='Greens',label='carbon',alpha=0.7,edgecolor='green',linewidth=2,vmin=0,vmax=15)
    ax.set_aspect(2)
    ax.figure.set_size_inches(30,30)
    ax_cbar = fig.colorbar(cbar,ax=ax,fraction=0.044, pad=0.04)
    ax_cbar.set_label('Mt C',fontsize=50)
    ax.set_ylabel('latitude',fontsize=50)
    ax.set_xlabel('longitude',fontsize=50)
    ax.set_title('Vegetation Carbon Gained from Reforestation in British Columbia',fontsize=50,pad=50)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    ax_cbar.ax.tick_params(labelsize=50)
    plt.savefig('bc_carbon_gained_2015.png')


    #plot tree cover gained from reforestation
    fig, ax = plt.subplots()
    cbar=plt.cm.ScalarMappable(cmap='Greens',norm=plt.Normalize(vmin=treeFrac_diff['treeFrac'].min(), vmax=treeFrac_diff['treeFrac'].max()))
    canada[canada.PRUID=='59'].plot(ax=ax, color='white', edgecolor='black')
    treeFrac_diff.plot(ax=ax,column='treeFrac',cmap='Greens',label='carbon',alpha=0.7,edgecolor='green',linewidth=2,vmin=0,vmax=15)
    ax.set_aspect(2)
    ax.figure.set_size_inches(30,30)
    ax_cbar = fig.colorbar(cbar,ax=ax,fraction=0.044, pad=0.04)
    ax_cbar.set_label('% Tree Cover',fontsize=50)
    ax.set_ylabel('latitude',fontsize=50)
    ax.set_xlabel('longitude',fontsize=50)
    ax.set_title('Tree Forest Cover Difference from Reforestation in British Columbia',fontsize=50,pad=50)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    ax_cbar.ax.tick_params(labelsize=50)
    plt.savefig('bc_tree_cover_gained_2015.png')

    #plot canadawide carbon
    fig, ax = plt.subplots()
    cbar=plt.cm.ScalarMappable(cmap='Greens',norm=plt.Normalize(vmin=canada_carbon['agb'].min(), vmax=canada_carbon['agb'].max()))
    canada.plot(ax=ax, color='white', edgecolor='black')
    canada_carbon.plot(ax=ax,column='agb',cmap='Greens',label='carbon',alpha=0.7,edgecolor='green',linewidth=2,vmin=0,vmax=15)
    ax.set_aspect(2)
    ax.figure.set_size_inches(30,30)
    ax_cbar = fig.colorbar(cbar,ax=ax,fraction=0.044, pad=0.04)
    ax_cbar.set_label('AGB Mt Carbon',fontsize=50)
    ax.set_ylabel('latitude',fontsize=50)
    ax.set_xlabel('longitude',fontsize=50)
    ax.set_title('Canada carbon stock in 2014 AGB',fontsize=50,pad=50)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    ax_cbar.ax.tick_params(labelsize=50)
    plt.savefig('carbon_2014.png')

    #plot canadawide carbon
    fig, ax = plt.subplots()
    cbar=plt.cm.ScalarMappable(cmap='Greens',norm=plt.Normalize(vmin=canada_carbon['agb'].min(), vmax=canada_carbon['agb'].max()))
    canada.plot(ax=ax, color='white', edgecolor='black')
    canada_carbon_2013.plot(ax=ax,column='agb',cmap='Greens',label='carbon',alpha=0.7,edgecolor='green',linewidth=2,vmin=0,vmax=15)
    ax.set_aspect(2)
    ax.figure.set_size_inches(30,30)
    ax_cbar = fig.colorbar(cbar,ax=ax,fraction=0.044, pad=0.04)
    ax_cbar.set_label('AGB Mt Carbon',fontsize=50)
    ax.set_ylabel('latitude',fontsize=50)
    ax.set_xlabel('longitude',fontsize=50)
    ax.set_title('Canada carbon stock in 2013 AGB',fontsize=50,pad=50)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    ax_cbar.ax.tick_params(labelsize=50)
    plt.savefig('carbon_2013.png')
main()