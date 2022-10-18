import plotly.graph_objects as go 
import pandas as pd
import xarray as xr

cesm_data = pd.read_csv('/Users/gclyne/thesis/data/cesm_data.csv').groupby('years').mean()
predicted_forest_carbon = pd.read_csv('/Users/gclyne/thesis/output/forest_carbon_observed.csv')

years = [x for x in range(1984,2019)]
fig = go.Figure()
fig.add_trace(go.Scatter(x=years, y=predicted_forest_carbon[0],
                    mode='lines',
                    name='lines'))
fig.add_trace(go.Scatter(x=years, y=df_merged_grouped['observed_bare'] ,
                    mode='markers',
                    name='markers'))
# fig.add_trace(go.Scatter(x=years, y=df_fin['# lai'],
#                     mode='lines+markers',
#                     name='lines+markers'))

fig.show()