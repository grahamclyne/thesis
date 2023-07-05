import os

from omegaconf import DictConfig
import hydra
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    for table in cfg.cmip_variable_map:
        for var in cfg.cmip_variable_map[table]:
            for variant in range(1,8):
                CMIP_TABLE=table
                CMIP_VARIABLE=var
                CMIP_EXPERIMENT = 'historical'
                CMIP_VARIANT = f'r{variant}i1p1f1'
                CMIP_SOURCE = 'CESM2'
                CMIP_NODE='aims3.llnl.gov'

                wget_string = f'wget http://esgf-node.llnl.gov/esg-search/wget\?project=CMIP6\&experiment_id={CMIP_EXPERIMENT}\&source_id={CMIP_SOURCE}\&variant_label={CMIP_VARIANT}\&table_id={CMIP_TABLE}\&variable_id={CMIP_VARIABLE}'

                print(table,var,wget_string)
                os.system(wget_string + f' -O {cfg.environment.cesm}/wget_{CMIP_VARIABLE}.txt')


                os.system(f'chmod +x {cfg.environment.cesm}/wget_{CMIP_VARIABLE}.txt')

                os.system(f'sh {cfg.environment.cesm}/wget_{CMIP_VARIABLE}.txt -s')
                # os.system(f'rm {config.CESM_PATH}/wget*.txt')

if __name__ == '__main__':
    main()