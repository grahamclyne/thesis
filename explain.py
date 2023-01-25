import torch
import pandas as pd
from model import Net
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="ann_config")
def main(cfg: DictConfig):
    model = Net(len(cfg.model.input),len(cfg.model.output))

    checkpoint = torch.load(f'ann_checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])


    from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation
    model.eval()
    ig = IntegratedGradients(model)
    final_input = pd.read_csv(f'data/cesm_data.csv')
    final_input = final_input.dropna(how='any')
    from pickle import load
    from other.constants import MODEL_TARGET_VARIABLES,MODEL_INPUT_VARIABLES
    scaler = load(open('/Users/gclyne/thesis/ann_scaler.pkl', 'rb'))
    final_input.loc[:,tuple(MODEL_INPUT_VARIABLES)] = scaler.transform(final_input[MODEL_INPUT_VARIABLES])
    final_input = final_input[MODEL_INPUT_VARIABLES]
    d = torch.tensor(final_input.to_numpy()).float()
    print(d.shape)
    ig.attribute(d,n_steps=1)

main()