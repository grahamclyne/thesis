import os
import other.constants as constants
import other.config as config

os.chdir(config.CESM_PATH)
for key in constants.CMIP_VARIABLES:
    wget_string = f'wget http://esgf-data.dkrz.de/esg-search/wget\?project=CMIP6\
\&experiment_id={constants.CMIP_EXPERIMENT}\
\&source_id={constants.CMIP_SOURCE}\
\&data_node=aims3.llnl.gov\
\&variant_label={constants.CMIP_VARIANT}\
\&table_id={key}\
\&variable_id={",".join(constants.CMIP_VARIABLES[key])}'


#how to verify each will give you four files? or the same amount of data? some variants are missing decades
    os.system(wget_string + f' -O wget_{key}.txt')
os.system('chmod +x wget*')

for key in constants.CMIP_VARIABLES:
    os.system(f'./wget_{key}.txt -s')
os.system('rm wget*')