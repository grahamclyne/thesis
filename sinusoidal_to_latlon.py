
import os
import re
import pyproj
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

USE_GDAL = False

def run(FILE_NAME):
    
    # Identify the data field.
    DATAFIELD_NAME = 'LST_Day_1km'

    if  USE_GDAL:    
        import gdal
        GRID_NAME = 'MODIS_Grid_16DAY_500m_VI'
        gname = 'HDF4_EOS:EOS_GRID:"{0}":{1}:{2}'.format(FILE_NAME,
                                                         GRID_NAME,
                                                         DATAFIELD_NAME)
        gdset = gdal.Open(gname)
        data = gdset.ReadAsArray().astype(np.float64)


        # Construct the grid.
        x0, xinc, _, y0, _, yinc = gdset.GetGeoTransform()
        nx, ny = (gdset.RasterXSize, gdset.RasterYSize)
        x = np.linspace(x0, x0 + xinc*nx, nx)
        y = np.linspace(y0, y0 + yinc*ny, ny)
        xv, yv = np.meshgrid(x, y)

        # In basemap, the sinusoidal projection is global, so we won't use it.
        # Instead we'll convert the grid back to lat/lons.
        sinu = pyproj.Proj("+proj=sinu +R=6371007.181 +nadgrids=@null +wktext")
        wgs84 = pyproj.Proj("+init=EPSG:4326") 
        lon, lat= pyproj.transform(sinu, wgs84, xv, yv)

        # Read the attributes.
        meta = gdset.GetMetadata()
        long_name = meta['long_name']        
        units = meta['units']
        _FillValue = np.float(meta['_FillValue'])
        scale_factor = np.float(meta['scale_factor'])
        valid_range = [np.float(x) for x in meta['valid_range'].split(', ')] 

        del gdset
    else:
        from pyhdf.SD import SD, SDC
        hdf = SD(FILE_NAME, SDC.READ)

        # Read dataset
        data2D = hdf.select(DATAFIELD_NAME)
        data = data2D[:,:].astype(np.double)

        # Read geolocation dataset from HDF-EOS2 dumper output.
        # Use the following command to generate latitude values in ASCII.
        # ~/Downloads/eos2dump -c1 MOD11A2.A2021321.h12v02.006.2021331131605.hdf > lat_MOD11A2
        GEO_FILE_NAME = 'lat_MOD11A2'
        lat = np.genfromtxt(GEO_FILE_NAME, delimiter=',', usecols=[0])
        lat = lat.reshape(data.shape)
        
        # Use the following command to generate longitude values in ASCII.
        # ~/Downloads/eos2dump -c2 MOD11A2.A2021321.h12v02.006.2021331131605.hdf > long_MOD11A2
        GEO_FILE_NAME = 'long_MOD11A2'
        lon = np.genfromtxt(GEO_FILE_NAME, delimiter=',', usecols=[0])
        lon = lon.reshape(data.shape)
        
        # Read attributes.
        attrs = data2D.attributes(full=1)
        lna=attrs["long_name"]
        long_name = lna[0]
        vra=attrs["valid_range"]
        valid_range = vra[0]
        fva=attrs["_FillValue"]
        _FillValue = fva[0]
        sfa=attrs["scale_factor"]
        scale_factor = sfa[0]        
        ua=attrs["units"]
        units = ua[0]
        print(data2D.attributes())
    
    invalid = np.logical_or(data > valid_range[1],
                            data < valid_range[0])
    invalid = np.logical_or(invalid, data == _FillValue)
    data[invalid] = np.nan
    data = data / scale_factor 
    data = np.ma.masked_array(data, np.isnan(data))



    #use ./tilemap3_linux is_k k fwd tp 61.3089, -121.2984
    
    for row in range(len(data)):
        for col in range(len(data[0])):
            if(row < 1042 or row > 1044 or col < 210 or col > 212):
                data[row][col] = np.nan
    #-121.2984,61.3089
    m = Basemap(projection='cyl', resolution='l',
                llcrnrlat=61.2, urcrnrlat = 61.4,
                llcrnrlon=-121.4, urcrnrlon = -121.2)                
    # m = Basemap(projection='cyl', resolution='l',
    #             llcrnrlat=60.5, urcrnrlat = 62.0,
    #             llcrnrlon=-122.0, urcrnrlon = -121.0)   
   # m.drawcoastlines(linewidth=0.2)
    m.drawparallels(np.arange(np.floor(np.min(lat)), np.ceil(np.max(lat)), .04),
                    labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(np.floor(np.min(lon)), np.ceil(np.max(lon)), .04),
                    labels=[0, 0, 0, 1])
    m.pcolormesh(lon, lat, data, latlon=True)
    cb = m.colorbar()
    cb.set_label(units)
    
    basename = os.path.basename(FILE_NAME)
    plt.title('{0}\n{1}'.format(basename, long_name))
    fig = plt.gcf()
    pngfile = "output{0}.py.png".format(basename)
    fig.savefig(pngfile)


if __name__ == "__main__":
    hdffile = 'MOD11A2.A2021321.h12v02.006.2021331131605.hdf'
    run(hdffile)



#https://hdfeos.org/zoo/LPDAAC/MOD13A1.A2007257.h09v05.006.2015167042211.hdf.py