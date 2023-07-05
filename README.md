to start virtualenv:
source env/bin/activate

to close:
deactivate

generate requirements file:
pipreqs .  (using pip freeze > requirements.txt will give all apckages in ENV, not just ones used in proj)

generate study area lat/lon bounds: 
python -m preprocessing.generate_shapefile_lats_lons
python -m preprocessing.compute_ecozone_coordinates

prep cmip data: 
python -m preprocessing.download_cmip_data
python -m preprocessing.generate_cmip_input_data

prep era data
python -m preprocessing.download_era_data
python -m preprocessing.generate_era_data


prep NFIS/NTEMS data:
manually download tree cover/harvest files from https://opendata.nfis.org/mapserver/nfis-change_eng.html
run gdal_warp.sh on cluster
python -m preprocessing.preprocess_nfis (or ./preprocess.sh on cluster)


create observed dataset: 
python -m preprocessing.combine_observational_data

prepare agb validation datasets:
python -m preprocess.process_biomass


train lstm:
./run_lstm.sh (on cluster, default is distributed cpus)

copy checkpoint (scaler and weights) from cluster, change run_name in conf file to match copied checkpoints

for analysis: 
python -m analysis.infer_lstm
python -m analysis.compare_observed_datasets
python -m analysis.harvest_scenario
python -m analysis.explain_lstm


other considerations: 
assuming CESM is geodesic and not planar when calculating m^2 area from lat lon

TODO 
k fold cross validation
split precipitation by season?
get optimal batch size for project
what is hist_interval?
include fHarvestToProduct analysis
expand LAMSE application
does normalizing precip remove its variability? how does normalizing affect this