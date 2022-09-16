TODO 
look at edge case for lat lon for loops 178, to -180
look into precipitation data differences
fix lat_range, lon_range

what is hist_interval
- COMPLETE DATA GEN SEPTEMBER 20, start SEPT 6

- convert epsg:3978 to wgs84 instead of current "bounds" hack
- fix year indexing of ERA data when selecting for reduction
- apply cloud mask to elevation model ? ----> when sum whole image collection, mask gives {'MSK': {'0': 480492399.4823542,
  '28': 98935.34509803921,
  '36': 1559988.196078431,
  '40': 13529832.705882354,
  '8': 0.16470588235294117}} (where bit of 1 is invalid, ie no invalid pixels)
- normalize each individual netcdf?

to start virtualenv:

source env/bin/activate

to close:

deactivate


to generate requirements file:

pipreq .  (using pip freeze > requirements.txt will give all apckages in ENV, not just ones used in proj)



geopandas and xarray have heavy dependency requirements, watch out


observable data: 
lai: float between 0 and ~11? 
temp: float in K 
