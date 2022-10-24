import plotly.graph_objects as go 
import pandas as pd
import xarray as xr

cesm_data = pd.read_csv('/Users/gclyne/thesis/data/cesm_data.csv').groupby('years').mean()
predicted_forest_carbon = pd.read_csv('/Users/gclyne/thesis/output/forest_carbon_observed.csv')
# cesm_data_grouped = cesm_data.groupby('year').mean()
predicted_forest_carbon.columns = ['year','cSoil','cCwd','cVeg','cLitter']
years = [x for x in range(1984,2020)]

def make_cesm_predict_plot(variable:str) -> go.Figure :
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=predicted_forest_carbon[variable],
                        mode='markers',
                        name='predicted',
                        marker=dict(color='brown')
                    ))
    fig.add_trace(go.Scatter(x=years, y=cesm_data[variable] ,
                        mode='lines',
                        name='cesm',
                        marker=dict(color='brown')
    ))

    fig.update_layout(title=f'{variable} estimated vs modeled')
    fig.update_xaxes(title_text="year")
    fig.update_yaxes(title_text="kg/m2")
    return fig
print(cesm_data)
print(predicted_forest_carbon)
for var in ['cVeg','cSoil','cCwd','cLitter']:
    fig = make_cesm_predict_plot(var)
    fig.write_image(f'/Users/gclyne/thesis/{var}_prediction_vs_cesm.png')
