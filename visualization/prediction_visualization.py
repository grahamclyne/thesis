import plotly.graph_objects as go 
import pandas as pd
import other.config as config
import datetime

def make_cesm_predict_comparison_plot(cesm_df:pd.DataFrame,predicted_df:pd.DataFrame,variable_name:str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=cesm_df,
                        mode='markers',
                        name='cesm',
                        marker=dict(color='brown')
                    ))
    fig.add_trace(go.Scatter(x=years, y=predicted_df ,
                        mode='lines',
                        name='predicted',
                        marker=dict(color='brown')
    ))

    fig.update_layout(title=f'{variable_name} estimated vs modeled')
    fig.update_xaxes(title_text="year")
    fig.update_yaxes(title_text="kg/m2")
    return fig


if __name__ == '__main__':
    cesm_data = pd.read_csv(f'{config.GENERATED_DATA}/cesm_data.csv').groupby('years').mean()
    predicted_forest_carbon = pd.read_csv('/Users/gclyne/thesis/output/forest_carbon_observed.csv')
    predicted_forest_carbon.columns = ['year'] + config.TARGET_VARIABLES
    years = [x for x in range(1984,2020)]
    for var in config.TARGET_VARIABLES:
        fig = make_cesm_predict_comparison_plot(cesm_data[var],predicted_forest_carbon[var],var)
        fig.write_image(f'{config.OUTPUT_PATH}/{var}_prediction_vs_cesm_{datetime.datetime.now().strftime("%y_%m_%d")}.png')
