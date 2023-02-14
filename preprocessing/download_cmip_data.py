import os
import preprocessing.config as config
CMIP_VARIABLES = {'Lmon':['nppRoot','nppWood','nppLeaf','rGrowth','rMaint','treeFracSecEver','treeFracSecDec','treeFracPrimEver','treeFracPrimDec','landCoverFrac']}

from hydra import initialize, compose

with initialize(version_base=None, config_path="../conf"):
    cfg = compose(config_name="ann_config")


for table in CMIP_VARIABLES:
    for var in CMIP_VARIABLES[table]:
        for variant in range(1,8):
            CMIP_TABLE=table
            CMIP_VARIABLE=var
            CMIP_EXPERIMENT = 'historical'
            CMIP_VARIANT = f'r{variant}i1p1f1'
            CMIP_SOURCE = 'CESM2'
            CMIP_NODE='aims3.llnl.gov'

            wget_string = f'wget http://esgf-node.llnl.gov/esg-search/wget\?project=CMIP6\&experiment_id={CMIP_EXPERIMENT}\&source_id={CMIP_SOURCE}\&variant_label={CMIP_VARIANT}\&table_id={CMIP_TABLE}\&variable_id={CMIP_VARIABLE}'

            print(table,var,wget_string)
            os.system(wget_string + f' -O {cfg.path.cesm}/wget_{CMIP_VARIABLE}.txt')


            os.system(f'chmod +x {cfg.path.cesm}/wget_{CMIP_VARIABLE}.txt')

            os.system(f'sh {cfg.path.cesm}/wget_{CMIP_VARIABLE}.txt -s')
            # os.system(f'rm {config.CESM_PATH}/wget*.txt')