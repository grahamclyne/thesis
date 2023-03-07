import pandas as pd

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    harvest_df = pd.read_csv(f'{cfg.environment.nfis_harvest_data}',header=None)
    harvest_df.columns = ['total_pixels'] + [x for x in range(1985,2016)] + ['sum','year','lat','lon']
    #convert harvests to percentages
    for year in range(1985,2016):
        harvest_df[year] = harvest_df[year] / harvest_df['total_pixels'] * 100
    harvest_df['lat'] = round(harvest_df['lat'],6)
    observed_data = pd.read_csv(f'{cfg.environment.observed_input}')
    observed_ts = pd.read_csv(f'{cfg.data}/observed_timeseries30_data.csv')
    observed_data = pd.merge(observed_data,harvest_df,  how='left', left_on=['lat','lon'], right_on = ['lat','lon'])
    observed_ts = pd.merge(observed_ts,harvest_df,  how='left', left_on=['lat','lon'], right_on = ['lat','lon']) 
    #for each year of harvest data, merge with the observed data and add the harvests to the treeFrac column
    for year in range(1985,2016):
        observed_data.loc[observed_data['year_x'].isin([x for x in range(year,2016)]),'treeFrac'] = (observed_data.loc[observed_data['year_x'].isin([x for x in range(year,2016)]),year]) + observed_data.loc[observed_data['year_x'].isin([x for x in range(year,2016)]),'treeFrac']
        print('before treefrac sum',observed_ts.loc[observed_ts['year_x'].isin([x for x in range(year,2016)]),'treeFrac'].sum())

        observed_ts.loc[observed_ts['year_x'].isin([x for x in range(year,2016)]),'treeFrac'] = (observed_ts.loc[observed_ts['year_x'].isin([x for x in range(year,2016)]),year]) + observed_ts.loc[observed_ts['year_x'].isin([x for x in range(year,2016)]),'treeFrac']
        print(observed_ts.loc[observed_ts['year_x'].isin([x for x in range(year,2016)]),year].sum())
        print('treefrac sum',observed_ts.loc[observed_ts['year_x'].isin([x for x in range(year,2016)]),'treeFrac'].sum())
        print(observed_ts['treeFrac'].sum())
        print(observed_ts.loc[observed_ts['year_x'].isin([x for x in range(year,2016)]),year])
    observed_data.rename(columns={'year_x':'year'},inplace=True)
    observed_data.to_csv(f'{cfg.environment.reforested_input}',index=False)
    observed_ts.to_csv(f'{cfg.data}/observed_reforest_ts.csv',index=False)
main()