import pandas as pd

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../conf", config_name="ann_config")
def main(cfg: DictConfig):

    harvest_df = pd.read_csv(f'{cfg.path.nfis_harvest_data}',header=None)
    harvest_df.columns = ['total_pixels'] + [x for x in range(1985,2016)] + ['sum','year','lat','lon']
    # provincial_dict = mapProvincialCoordinates()

    #convert harvests to percentages
    for year in range(1985,2016):
        harvest_df[year] = harvest_df[year] / harvest_df['total_pixels'] * 100
    
    harvest_df['lat'] = round(harvest_df['lat'],6)
    observed_data = pd.read_csv(f'{cfg.path.observed_input}')
    out_df = pd.DataFrame()
    observed_data = pd.merge(observed_data,harvest_df,  how='left', left_on=['lat','lon'], right_on = ['lat','lon'])
    print(observed_data['lai'].sum())
    #for each year of harvest data, merge with the observed data and add the harvests to the treeFrac column
    for year in range(1985,2016):
        # print((observed_data.loc[observed_data['year_x'].isin([x for x in range(year,2016)]),year]))
        # print(observed_data.loc[observed_data['year_x'].isin([x for x in range(year,2016)]),'treeFrac'])
        # print((observed_data.loc[observed_data['year_x'].isin([x for x in range(year,2016)]),year]) + observed_data.loc[observed_data['year_x'].isin([x for x in range(year,2016)]),'treeFrac'])
        observed_data.loc[observed_data['year_x'].isin([x for x in range(year,2016)]),'treeFrac'] = (observed_data.loc[observed_data['year_x'].isin([x for x in range(year,2016)]),year]) + observed_data.loc[observed_data['year_x'].isin([x for x in range(year,2016)]),'treeFrac']
        observed_data.loc[observed_data['year_x'].isin([x for x in range(year,2016)]),'lai'] = (observed_data.loc[observed_data['year_x'].isin([x for x in range(year,2016)]),year])/100  + observed_data.loc[observed_data['year_x'].isin([x for x in range(year,2016)]),'lai']

        # year_data = observed_data[observed_data['year'] == year].reset_index(drop=True)
        # new_df = pd.merge(year_data,harvest_df,  how='left', left_on=['lat','lon'], right_on = ['lat','lon'])
        # new_column = (new_df[year]*100) + new_df['treeFrac']
        # year_data['treeFrac'] = new_column
        # out_df = pd.concat([out_df,year_data])
        # print(observed_data.loc[observed_data['year_x'].isin([x for x in range(year,2016)]),['treeFrac']])
    print(observed_data['lai'].sum())
    observed_data.rename(columns={'year_x':'year'},inplace=True)
    observed_data.to_csv(f'{cfg.path.reforested_input}',index=False)

    #calculate each province's total harvest by year - make another dict with a list of the harvests for each year
        
main()