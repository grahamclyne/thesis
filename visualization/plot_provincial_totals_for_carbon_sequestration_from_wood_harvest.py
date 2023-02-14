import pandas as pd
import geopandas
import pickle
from hydra import initialize, compose
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.utils import getArea
with initialize(version_base=None, config_path="../conf"):
    cfg = compose(config_name="ann_config")

no_reforestation = pd.read_csv('data/hybrid_no_reforest.csv')
total_reforestation = pd.read_csv('data/hybrid_reforest.csv')
total_reforestation = total_reforestation[(total_reforestation['year'] > 1984) & (total_reforestation['year'] < 2015)].reset_index(drop=True)
no_reforestation = no_reforestation[(no_reforestation['year'] > 1984) & (no_reforestation['year'] < 2015)].reset_index(drop=True)
canada = geopandas.read_file('/Users/gclyne/thesis/data/shapefiles/lpr_000b21a_e/lpr_000b21a_e.shp')

total_reforestation['agb'] = total_reforestation['cLeaf'] + total_reforestation['cStem'] + total_reforestation['cOther']
no_reforestation['agb'] = no_reforestation['cLeaf'] + no_reforestation['cStem'] + no_reforestation['cOther']
total_reforestation['area'] = total_reforestation.apply(lambda x: getArea(x['lat'],x['lon'],cfg),axis=1)
no_reforestation['area'] = no_reforestation.apply(lambda x: getArea(x['lat'],x['lon'],cfg),axis=1)
total_reforestation['agb'] = total_reforestation['agb'] * total_reforestation['area'] / 1e9 #to megatonnes
no_reforestation['agb'] = no_reforestation['agb'] * no_reforestation['area'] / 1e9 # kg to megatonnes
# provincial_dict = mapProvincialCoordinates(cfg)
with open('/Users/gclyne/thesis/provincial_coords.pickle', 'rb') as handle:
    provincial_dict = pickle.load(handle) 

bc_lats = pd.DataFrame([x[0] for x in provincial_dict['British Columbia / Colombie-Britannique']])
bc_lons = pd.DataFrame([x[1] for x in provincial_dict['British Columbia / Colombie-Britannique']])
bc = pd.concat([bc_lats,bc_lons],axis=1)
bc.columns = ['lon','lat']
bc.lat = round(bc.lat,6)
re_bc = pd.merge(total_reforestation,bc,how='inner',left_on=['lat','lon'],right_on=['lat','lon'])
no_bc = pd.merge(no_reforestation,bc,how='inner',left_on=['lat','lon'],right_on=['lat','lon'])


provincial_totals = {}

for province in provincial_dict.keys():
    lats = pd.DataFrame([x[0] for x in provincial_dict[province]])
    lons = pd.DataFrame([x[1] for x in provincial_dict[province]])
    df = pd.concat([lats,lons],axis=1)
    df.columns = ['lon','lat']
    df.lat = round(df.lat,6)
    re_df = pd.merge(total_reforestation,df,how='inner',left_on=['lat','lon'],right_on=['lat','lon'])
    no_df = pd.merge(no_reforestation,df,how='inner',left_on=['lat','lon'],right_on=['lat','lon'])
    provincial_totals[province] = [re_df[re_df['year'] == 2014]['agb'].sum() ,no_df[no_df['year'] == 2014]['agb'].sum()]

restored = [x[0] for x in provincial_totals.values()]
no_restored = [x[1] for x in provincial_totals.values()]
labels = provincial_totals.keys()
labels = ['ON','NS','QC','NB','PEI','NF','BC','AB','SK','MB','YK','NT']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
difference = [round((x[0] - x[1]),2) for x in zip(restored,no_restored)]
fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, restored, width, label='restored forest')
# rects2 = ax.bar(x + width/2, no_restored, width, label='not restored')
diff1 = ax.bar(x,difference,width,label='Carbon sequestered')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Above-Ground Biomass (Mt C)')
ax.set_title('Carbon sequestration by province for reforestation from 1985-2014')
ax.set_xticks(x, labels)
ax.legend()
print(sum(difference))
ax.bar_label(diff1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.tight_layout()
plt.savefig('carbon_sequestration_by_province_2014_totals.png')





#plot yearly view of carbon sequestration
fig, ax = plt.subplots()

ax.plot(re_bc.groupby('year')['agb'].sum() - no_bc.groupby('year')['agb'].sum())
ax.set_ylabel('Carbon (Mt)')
plt.title('British Columbia')
plt.savefig("bc_yearly_carbon_sequestration.png")

#plot yearly view of carbon sequestration
fig, ax = plt.subplots()
carbon_diff = (total_reforestation.groupby('year')['agb'].sum() - no_reforestation.groupby('year')['agb'].sum())
fin_carbon_diff = [x - y for x,y in zip(carbon_diff.iloc[1:],carbon_diff)]
# fin_carbon_diff = [x if x > 0 else 0 for x in fin_carbon_diff]
# print(carbon_diff,carbon_diff.iloc[0])
ax.bar([x for x in range(1986,2015)],fin_carbon_diff)
ax.set_ylabel('Carbon (Mt)')
plt.title('Canada-wide Carbon Sequestration from Reforestation')
plt.savefig("canada_yearly_carbon_sequestration.png")
# pd.set_option('display.max_columns', 500)
# print(total_reforestation.groupby('year').describe())
# print(no_reforestation.groupby('year').describe())