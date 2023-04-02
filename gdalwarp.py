import os 
# gdalwarp -t_srs EPSG:4326 -s_srs EPSG:3978 /Users/gclyne/Downloads/CA_forest_harvest_years2recovery/CA_forest_harvest_years2recovery.tif output_raster.tif
# for file in os.listdir('~/scratch'):
#     if('forest' in file):
#         for f in os.listdir('data/NFIS/'+file):
#             if(f.endswith('.tif')):
#                 print(f)
#                 os.system(f'gdalwarp -t_srs EPSG:4326 -s_srs EPSG:3978 ~/scratch{f} {f}_reprojected_to_4326.tif')
#                 # os.execute('gdalwarp -t_srs EPSG:4326 data/NFIS/'+file+'/'+f+' data/NFIS/'+file+'/'+f[:-4]+'_4326.tif')

for file in os.listdir('/home/gclyne/scratch'):
    if('forest' in file and file.endswith('.tif')):
        os.system(f'gdalwarp -t_srs EPSG:4326 -s_srs EPSG:3978 ~/scratch{file} reprojected_4326_{file}')

