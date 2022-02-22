#landsatxplore search --dataset LANDSAT_TM_C1 --location 53.54 -113.49 --start 1995-01-01 --end 1995-12-31 --username grahamclyne --password eros45eraseris



import json
from landsatxplore.api import API
from landsatxplore.earthexplorer import EarthExplorer

ee = EarthExplorer('grahamclyne', 'eros45eraseris')
# Initialize a new API instance and get an access key
api = API('grahamclyne', 'eros45eraseris')

# Search for Landsat TM scenes
scenes = api.search(
    dataset='landsat_tm_c1',
    latitude=45.5017,
    longitude=-73.5673,
    start_date='1995-01-01',
    end_date='1995-10-01',
    max_cloud_cover=10
)

print(f"{len(scenes)} scenes found.")

# Process the result

ee.download(scenes[0]['landsat_scene_id'], output_dir=scenes[0]['landsat_scene_id'] + '_data')

ee.logout()
api.logout()