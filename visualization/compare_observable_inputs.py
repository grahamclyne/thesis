import plotly.graph_objects as go 
import pandas as pd
import preprocessing.config as config
import datetime
import hydra
from omegaconf import DictConfig


def make_cesm_predict_comparison_plot(cesm_df:pd.DataFrame,predicted_df:pd.DataFrame,variable_name:str,years:list) -> go.Figure:
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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    cesm_data = pd.read_csv(f'{cfg.data}/cesm_data.csv').groupby('# year').mean()
    observed_input = pd.read_csv(cfg.environment.path.observed_input)
    observed_input = observed_input[cfg.model.input]
    cesm_data = cesm_data[cfg.model.input]
    years = [x for x in range(1984,2016)]
    for var in cfg.model.input:
        fig = make_cesm_predict_comparison_plot(cesm_data[var],observed_input[var],var,years)
        fig.show()
        # fig.write_image(f'{config.OUTPUT_PATH}/{var}_prediction_vs_cesm_{datetime.datetime.now().strftime("%y_%m_%d")}.png')

main()