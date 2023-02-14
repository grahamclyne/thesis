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


class RegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units,cfg):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = cfg.model.params.num_layers
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=len(cfg.model.output))

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.
        out = out.reshape(-1,len(self.cfg.model.output))
        return out
    

@hydra.main(version_base=None, config_path="conf",config_name='config')
def main(cfg: DictConfig):
    model = RegressionLSTM(num_sensors=len(cfg.model.input), hidden_units=cfg.model.params.hidden_units,cfg=cfg)
    torch.set_num_threads(8)

    # Define Loss, Optimizer
    loss_function = nn.MSELoss()
    wandb.init(project="rnn-land-carbon", entity="gclyne",config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    from transformer.transformer_model import CMIPTimeSeriesDataset
    cesm_df = pd.read_csv(f'{cfg.data}/cesm_transformer_data.csv')
    # cesm_df = cesm_df[cesm_df['# year'] < 1985]
    cesm_df.rename(columns={'# year':'year'},inplace=True)
    #need to reshape here for data scaling
    # split data into 6 chunks, use 4 for training, 1 for validation, 1 for hold out
    # chunk_size = len((np.array_split(cesm_df, 6, axis=0))[0])
    chunk_size = len(cesm_df)//6
    cesm_df = cesm_df[cfg.model.input + cfg.model.output + cfg.model.id]
    cesm_df = cesm_df.to_numpy()
    cesm_df = cesm_df.reshape(-1,30,len(cfg.model.input + cfg.model.output + cfg.model.id))
    np.random.shuffle(cesm_df)
    cesm_df = cesm_df.reshape(-1,len(cfg.model.input + cfg.model.output + cfg.model.id))
    print(cesm_df)
    cesm_df = pd.DataFrame(cesm_df)
    cesm_df.columns = cfg.model.input + cfg.model.output + cfg.model.id
    print(cesm_df.head())
    hold_out = cesm_df[-chunk_size:]
    train_ds = cesm_df[:chunk_size*5]
    #CHUNK SIZE IS WRONG ---- keeps all northern latitudes for hold out and rest for train ds, need to shuffle
    print(cesm_df.columns)
    #fix that you are modifying targets here too
    scaler = preprocessing.StandardScaler().fit(train_ds.loc[:,cfg.model.input])
    out_scaler = preprocessing.StandardScaler().fit(train_ds.loc[:,cfg.model.output])
    hold_out_scaler = preprocessing.StandardScaler().fit(hold_out.loc[:,cfg.model.input])
    hold_out_out_scaler = preprocessing.StandardScaler().fit(hold_out.loc[:,cfg.model.output])

    hold_out.loc[:,cfg.model.input] = hold_out_scaler.transform(hold_out.loc[:,cfg.model.input])
    train_ds.loc[:,cfg.model.input] = scaler.transform(train_ds.loc[:,cfg.model.input])
    hold_out.loc[:,cfg.model.output] = hold_out_out_scaler.transform(hold_out.loc[:,cfg.model.output])
    train_ds.loc[:,cfg.model.output] = out_scaler.transform(train_ds.loc[:,cfg.model.output])

    dump(scaler, open(f'{cfg.environment.path.checkpoint}/lstm_scaler.pkl','wb'))
    dump(out_scaler, open(f'{cfg.environment.path.checkpoint}/lstm_output_scaler.pkl','wb'))
    hold_out = CMIPTimeSeriesDataset(hold_out,cfg.model.params.seq_len,len(cfg.model.input + cfg.model.output + cfg.model.id),cfg)
    train_ds = CMIPTimeSeriesDataset(train_ds,cfg.model.params.seq_len,len(cfg.model.input + cfg.model.output + cfg.model.id),cfg)
    train,validation = torch.utils.data.random_split(train_ds, [int((chunk_size/30)*4), int(chunk_size/30)], generator=torch.Generator().manual_seed(0))

    train_ldr = torch.utils.data.DataLoader(train,batch_size=cfg.model.params.batch_size,shuffle=True)
    validation_ldr = torch.utils.data.DataLoader(validation,batch_size=cfg.model.params.batch_size,shuffle=True)
    hold_out_ldr = torch.utils.data.DataLoader(hold_out,batch_size=cfg.model.params.batch_size,shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.params.lr)


    for epoch_count in range(cfg.model.params.epochs):
            total_start = time.time()

            print('Epoch:',epoch_count)
            train_loss = 0
            model.train()
            for src,tgt,id in train_ldr:
                optimizer.zero_grad() #clears old gradients from previous steps 
                pred_y = model(src.float())
                loss = loss_function(pred_y, tgt.float())
                wandb.log({"training_loss": loss})
                loss.backward() #compute gradient
                optimizer.step() #take step based on gradient
                train_loss += loss.item() 
            wandb.log({'total_training_loss':train_loss})

            model.eval()
            valid_loss = 0
            for src,tgt,id in validation_ldr:
                pred_y = model(src.float())

                loss = loss_function(pred_y,tgt.float())
                valid_loss += loss.item()
                wandb.log({"validation_loss": loss})
        
            wandb.log({'total_valid_loss':valid_loss})
            hold_out_loss=0
            total_predictions = []
            total_targets = []
            for src,tgt,id in hold_out_ldr:
                pred_y = model(src.float())
                loss = loss_function(pred_y,tgt.float())
                hold_out_loss += loss.item() 
                total_predictions.append(pred_y.detach().numpy())
                total_targets.append(tgt.detach().numpy())
                wandb.log({"hold_out_loss": loss})
                print(id[0])
                print(hold_out_out_scaler.inverse_transform(pred_y.detach().numpy())[0])
                print(hold_out_out_scaler.inverse_transform(tgt.detach().numpy())[0])
            epoch_time = time.time() - total_start
            wandb.log({'epoch time':epoch_time})

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, f'{cfg.environment.path.checkpoint}/lstm_checkpoint.pt')
main()