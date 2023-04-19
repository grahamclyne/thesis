import torch.nn as nn
import torch
import numpy as np 
from omegaconf import DictConfig




def lat_adjusted_mse(y_true, y_pred,lat):
    lat_factor = np.cos(np.deg2rad(lat))
    mse = torch.mean(torch.square(y_true - y_pred),-1)
    lat_mse = mse * lat_factor
    return torch.mean(lat_mse)


class CMIPTimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data:np.ndarray,seq_len:int,num_features:int,cfg:DictConfig):
        self.data = data.to_numpy().reshape(-1,seq_len,num_features)
        self.cfg = cfg
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # if(idx > len(self.data) - 5):
        #     raise StopIteration
        source = self.data[idx,:,:len(self.cfg.model.input)]
        target = self.data[idx,-1,-(len(self.cfg.model.output)+len(self.cfg.model.id)):-len(self.cfg.model.id)]
        id = self.data[idx,-1,-len(self.cfg.model.id):]
        return source,target,id

class RegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units,cfg):
        super().__init__()
        self.num_sensors = num_sensors  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = cfg.model.num_layers
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
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units,device=x.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units,device=x.device).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.
        out = out.reshape(-1,len(self.cfg.model.output))
        return out
    
