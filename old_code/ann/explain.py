import torch
import pandas as pd
from ann.ann_model import Net
import hydra
from omegaconf import DictConfig
from pickle import load
from captum.attr import LRP
from captum.attr._utils.lrp_rules import (
    Alpha1_Beta0_Rule,
    EpsilonRule,
    GammaRule,
    IdentityRule,
    PropagationRule)
@hydra.main(version_base=None, config_path="../conf", config_name="ann_config")
def main(cfg: DictConfig):
    model = Net(len(cfg.model.input),len(cfg.model.output))

    checkpoint = torch.load(f'checkpoint/ann_checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    
    lrp = LRP(model)
    final_input = pd.read_csv(f'data/cleaned_observed_ann_input.csv')
    final_input = final_input.dropna(how='any')

    scaler = load(open('/Users/gclyne/thesis/checkpoint/ann_scaler.pkl', 'rb'))
    final_input.loc[:,tuple(cfg.model.input)] = scaler.transform(final_input[cfg.model.input])
    final_input = final_input[cfg.model.input].to_numpy()
    print(lrp.model)
    lrp.model.add_module('batchnorm1',torch.nn.modules.batchnorm.BatchNorm1d(100))
    lrp.model.batchnorm1.rule = IdentityRule()

    attr = lrp.attribute(torch.tensor(final_input).float(),target=2) #target 2 is cVeg
    print(attr.mean(axis=0))
main()