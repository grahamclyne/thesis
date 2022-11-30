import os
import other.config as config
CMIP_TABLE='Lmon'
CMIP_VARIABLE='shrubFrac'
CMIP_EXPERIMENT = 'historical'
CMIP_VARIANT = 'r8i1p1f1'
CMIP_SOURCE = 'CESM2'
CMIP_NODE='aims3.llnl.gov'

wget_string = f'wget http://esgf-data.dkrz.de/esg-search/wget\?project=CMIP6\
\&experiment_id={CMIP_EXPERIMENT}\
\&source_id={CMIP_SOURCE}\
\&data_node={CMIP_NODE}\
\&variant_label={CMIP_VARIANT}\
\&table_id={CMIP_TABLE}\
\&variable_id={CMIP_VARIABLE}'
os.system(wget_string + f' -O {config.CESM_PATH}/wget_{CMIP_VARIABLE}.txt')


os.system(f'chmod +x {config.CESM_PATH}/wget*')

os.system(f'sh {config.CESM_PATH}/wget_{CMIP_VARIABLE}.txt -s')
os.system(f'rm {config.CESM_PATH}/wget*.txt')