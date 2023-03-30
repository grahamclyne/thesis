import torch.nn as nn
import torch
import pandas as pd
from omegaconf import DictConfig
import hydra
from sklearn import preprocessing
import omegaconf
from pickle import dump
import time
import wandb
import torch
import numpy as np
from lstm_model import RegressionLSTM
from train_lstm_dist import get_training_data
@hydra.main(version_base=None, config_path="conf",config_name='config')
def main(cfg: DictConfig):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RegressionLSTM(num_sensors=len(cfg.model.input), hidden_units=cfg.model.hidden_units,cfg=cfg).cuda()
    # Define Loss, Optimizer
    loss_function = nn.MSELoss().cuda()
    wandb.init(project="land-carbon-gpu", entity="gclyne",config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate) 
    run = 'single'
    train_sampler,train_loader,validation_sampler,validation_loader,test_sampler,test_loader = get_training_data(cfg,run)

    for epoch_count in range(cfg.model.epochs):
        total_start = time.time()
        train_sampler.set_epoch(epoch_count)
        validation_sampler.set_epoch(epoch_count)
        test_sampler.set_epoch(epoch_count)
        print('Epoch:',epoch_count)
        train_loss = 0
        model.train()
        for src,tgt,_ in train_loader:
            optimizer.zero_grad() #clears old gradients from previous steps 
            src = src.cuda()
            tgt = tgt.cuda()
            pred_y = model(src.float())
            loss = loss_function(pred_y, tgt.float())
            wandb.log({"training_loss": loss})
            loss.backward() #compute gradient
            optimizer.step() #take step based on gradient
            train_loss += loss.item() 
        wandb.log({'total_training_loss':train_loss})

        model.eval()
        valid_loss = 0
        for src,tgt,_ in validation_loader:
            src = src.cuda()
            tgt = tgt.cuda()
            pred_y = model(src.float())
            loss = loss_function(pred_y,tgt.float())
            valid_loss += loss.item()
            wandb.log({"validation_loss": loss})
        wandb.log({'total_valid_loss':valid_loss})

        epoch_time = time.time() - total_start
        wandb.log({'epoch time':epoch_time})
        torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, f'{cfg.environment.checkpoint}/lstm_checkpoint_{wandb.run.name}.pt')

    hold_out_loss=0
    total_predictions = []
    total_targets = []
    for src,tgt,_ in test_loader:
        src = src.cuda()
        tgt = tgt.cuda()
        pred_y = model(src.float())
        loss = loss_function(pred_y,tgt.float())
        hold_out_loss += loss.item() 
        total_predictions.append(pred_y.detach().numpy())
        total_targets.append(tgt.detach().numpy())
        wandb.log({"hold_out_loss": loss})
        print(pred_y.detach().numpy())
        print(tgt.detach().numpy())


main()
