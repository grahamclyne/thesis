# actual size of managed forest 4406620292520.00000000000
# size that gridded cells give 4335325524787.99
#gridded gives  71294767732.00977 less that actual, approx 1.61% less area
4335325524787.99 * .58 / 10000

#how much forested land does cesm say? does nfis say? 
from utils import getArea,getCoordinates,readCoordinates
cesm_data = pd.read_csv('/Users/gclyne/thesis/data/cesm_data.csv')
ordered_latitudes = readCoordinates('grid_latitudes.csv',is_grid_file=True)
ordered_longitudes = readCoordinates('grid_longitudes.csv',is_grid_file=True)
cesm_total_forested = 0

nfis_tree_cover = pd.read_csv('/Users/gclyne/thesis/data/generated_observable_data/nfis_tree_cover_data.csv')
nfis_grouped = nfis_tree_cover.groupby('year').mean()

tree_frac_nfis_totals = []
for i,row in nfis_grouped.iterrows():
    tree_frac_nfis_totals.append((row['observed_shrub_bryoid_herb'] / 100) * 4335325524787.99 / 10000000 )
tree_frac_nfis_totals



# for i,row in cesm_data.iterrows():
#     print(row['latitude'])
#     if(row['latitude'] == 43.82199096679688):
#         row['latitude'] = 43.821990966796875
#     cesm_total_forested += (getArea(*getCoordinates((row['latitude'],row['longitude']),ordered_latitudes,ordered_longitudes)) * row['treeFrac'])
# print(cesm_total_forested)

cesm_data_grouped = cesm_data.groupby('years').mean()
tree_frac_cesm_totals = []
for i,row in cesm_data_grouped.iterrows():
    # print(row['treeFrac'])
    tree_frac_cesm_totals.append(((row['grassFrac'] + row['cropFrac'] + row['shrubFrac']) / 100) * 4335325524787.99 / 10000000 )
tree_frac_cesm_totals


fig = go.Figure()
reported = [54000 for s in years]
fig.add_trace(go.Scatter(x=years, y=tree_frac_cesm_totals,
                    mode='markers',
                    name='cesm grass coverage,'
                ))
fig.add_trace(go.Scatter(x=years, y=tree_frac_nfis_totals,
                    mode='lines',
                    name='nfis',
))
fig.add_trace(go.Scatter(x=years, y=reported,
                    mode='lines',
                    name='reported',
))
fig.update_layout(legend_title_text = "grass/shrub/crop fracton covers",title='grass/shrub/crop fraction coverage comparisons')
fig.update_xaxes(title_text="year")
fig.update_yaxes(title_text="kha")

fig.show()