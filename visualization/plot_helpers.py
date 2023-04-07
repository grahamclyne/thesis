import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
import seaborn as sns
from preprocessing.utils import getGeometryBoxes,readCoordinates
def plotCountryWideGridded(df_list,variable,titles=None,main_title=None):
        canada = gpd.read_file(f'data/shapefiles/lpr_000b16a_e/lpr_000b16a_e.shp')
        canada  = canada.to_crs('4326')
        f, axes = plt.subplots(figsize=(40, 10),nrows=1,ncols=len(df_list))
        if(len(df_list) > 1):
            axes = axes.flatten()
        else:
             axes = [axes]
        norm = mpl.colors.Normalize(df_list[0][variable].min(),df_list[0][variable].max(),clip=True)
        for ax_index in range(0,len(axes)):
            canada.plot(ax=axes[ax_index],alpha=0.1)
            ax = df_list[ax_index].plot(ax=axes[ax_index],column=variable,norm=norm,cmap='hot')
            ax.set_xlabel('Longitude',fontsize=10)
            ax.set_ylabel('Latitude',fontsize=10)
            # x = mpl.image.AxesImage(ax=axes[ax_index])
            axes[ax_index].title.set_text(titles[ax_index])
            axes[ax_index].title.set_fontsize(30)
        m = plt.cm.ScalarMappable(cmap='hot')
        m.set_array(df_list[0][variable])
        cbar = plt.colorbar(m,fraction=0.026, pad=0.04)
        cbar.ax.set_ylabel(f'{variable} (Mt C)',fontsize=20)
        f.tight_layout()
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=1.1,
                            wspace=0.1,
                            hspace=0)
        f.suptitle(main_title,fontsize=40)
        plt.show()

def plotNSTransect(lon):
    forest_coordinates = readCoordinates(f'data/ecozones_coordinates.csv',is_grid_file=False)
    coords = [x for x in forest_coordinates if x[1] == lon]
    x = pd.DataFrame(list(zip(*coords))).T
    x.columns = ['lat','lon']
    df = gpd.GeoDataFrame([],geometry=getGeometryBoxes(x))
    canada = gpd.read_file(f'data/shapefiles/lpr_000b16a_e/lpr_000b16a_e.shp')
    canada  = canada.to_crs('4326')
    f, axes = plt.subplots(figsize=(10, 10))
    canada.plot(ax=axes,alpha=0.2)
    df.plot(ax=axes)
    plt.title(f'Forest North-South Transect at Longitude {lon}',fontsize=15)
    plt.savefig(f'NS_transect_{lon}.png')