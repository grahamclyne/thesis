import pandas as pd
import other.config as config

observed = pd.read_csv(f'{config.DATA_PATH}/finalized_output.csv')
cesm = pd.read_csv(f'{config.DATA_PATH}/cesm_data.csv')

cesm = cesm.rename(columns={'# year':'year'})

yearly_era = observed.groupby('year').mean()
yearly_cesm = cesm.groupby('# year').mean()


yearly_era.to_csv(f'{config.DATA_PATH}/observed_yearly_groupby.csv')
yearly_cesm.to_csv(f'{config.DATA_PATH}/cesm_yearly_groupby.csv')