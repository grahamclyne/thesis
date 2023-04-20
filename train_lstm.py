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
from lstm_model import RegressionLSTM,CMIPTimeSeriesDataset,lat_adjusted_mse
from train_lstm_dist import split_data
from torcheval.metrics import R2Score

def get_training_data(cfg,run):
    cesm_df = pd.read_csv(f'{cfg.data}/timeseries_cesm_training_data_30.csv')
    ecozones_coords = pd.read_csv(f'{cfg.data}/ecozones_coordinates.csv')
    ecozones_coords = ecozones_coords[ecozones_coords['zone'].isin(['Boreal Cordillera','Boreal PLain', 'Boreal Shield'])]
    merged = pd.merge(cesm_df,ecozones_coords,on=['lat','lon'],how='inner')

    merged = merged.drop(columns=['zone'])
    merged = merged[cfg.model.input + cfg.model.output + cfg.model.id]

    train_ds,val_ds,test_ds = split_data(merged)
    scaler = preprocessing.StandardScaler().fit(train_ds.loc[:,cfg.model.input])
    train_ds.loc[:,cfg.model.input] = scaler.transform(train_ds.loc[:,cfg.model.input])
    val_ds.loc[:,cfg.model.input] = scaler.transform(val_ds.loc[:,cfg.model.input])
    test_ds.loc[:,cfg.model.input] = scaler.transform(test_ds.loc[:,cfg.model.input])

    if run != None:
        dump(scaler, open(f'{cfg.environment.checkpoint}/lstm_scaler_{run.name}_gpu.pkl','wb'))

    train_ds = CMIPTimeSeriesDataset(train_ds,cfg.model.seq_len,len(cfg.model.input + cfg.model.output + cfg.model.id),cfg)
    val_ds = CMIPTimeSeriesDataset(val_ds,cfg.model.seq_len,len(cfg.model.input + cfg.model.output + cfg.model.id),cfg)
    test_ds = CMIPTimeSeriesDataset(test_ds,cfg.model.seq_len,len(cfg.model.input + cfg.model.output + cfg.model.id),cfg)

    train_ldr = torch.utils.data.DataLoader(train_ds,batch_size=cfg.model.batch_size,shuffle=True)
    validation_ldr = torch.utils.data.DataLoader(val_ds,batch_size=cfg.model.batch_size,shuffle=True)
    test_ldr = torch.utils.data.DataLoader(test_ds,batch_size=cfg.model.batch_size,shuffle=True)

    # hold_out_ldr = torch.utils.data.DataLoader(hold_out,batch_size=cfg.model.batch_size,shuffle=True)
    return train_ldr,validation_ldr,test_ldr



@hydra.main(version_base=None, config_path="conf",config_name='config')
def main(cfg: DictConfig):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RegressionLSTM(num_sensors=len(cfg.model.input), hidden_units=cfg.model.hidden_units,cfg=cfg).cuda()
    # Define Loss, Optimizer

    loss_function = lat_adjusted_mse
    run = wandb.init(project="land-carbon-gpu", entity="gclyne",config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)
    train_loader,validation_loader,test_loader = get_training_data(cfg,run)

    for epoch_count in range(cfg.model.epochs):
        total_start = time.time()
        print('Epoch:',epoch_count)
        batch_train_loss = 0
        model.train()
        for src,tgt,id in train_loader:
            optimizer.zero_grad() #clears old gradients from previous steps 
            src = src.cuda()
            tgt = tgt.cuda()
            pred_y = model(src.float())
            loss = loss_function(pred_y, tgt.float(),id[:,1].cuda())
            wandb.log({"training_loss": loss})
            loss.backward() #compute gradient
            optimizer.step() #take step based on gradient
            train_loss += loss.item()
        metric = R2Score()
        metric.update(pred_y.cpu().detach().numpy(),tgt.cpu().detach().numpy())
        wandb.log({'train_r2_score':metric.compute()})

        wandb.log({'batch_training_loss':batch_train_loss})

        model.eval()
        valid_loss = 0
        for src,tgt,id in validation_loader:
            src = src.cuda()
            tgt = tgt.cuda()
            pred_y = model(src.float())
            loss = loss_function(pred_y,tgt.float(),id[:,1].cuda())
            valid_loss += loss.item()
            wandb.log({"validation_loss": loss})
        wandb.log({'batch_valid_loss':valid_loss})
        metric = R2Score()
        metric.update(pred_y.cpu().detach().numpy(),tgt.cpu().detach().numpy())
        wandb.log({'validation_r2_score':metric.compute()})
        torch.save({'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(),}, f'{cfg.environment.checkpoint}/lstm_checkpoint_{wandb.run.name}_gpu.pt')
        epoch_time = time.time() - total_start
        wandb.log({'epoch time':epoch_time})

    hold_out_loss=0
    total_predictions = []
    total_targets = []
    for src,tgt,id in test_loader:
        src = src.cuda()
        tgt = tgt.cuda()
        pred_y = model(src.float())
        loss = loss_function(pred_y,tgt.float(),id[:,1].cuda())
        hold_out_loss += loss.item() 
        total_predictions.append(pred_y.cpu().detach().numpy())
        total_targets.append(tgt.cpu().detach().numpy())
        wandb.log({"hold_out_loss": loss})
        print(pred_y.cpu().detach().numpy())
        print(tgt.cpu().detach().numpy())
    metric = R2Score()
    metric.update(pred_y.cpu().detach().numpy(),tgt.cpu().detach().numpy())
    wandb.log({'test_r2_score':metric.compute()})
    wandb.log({'total_hold_out_loss':hold_out_loss})


main()
