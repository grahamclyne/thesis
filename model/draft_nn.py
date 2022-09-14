import xarray as xr
import geopandas 
from shapely.geometry import mapping 
import numpy as np
import torch as T
import torch.nn as nn
import matplotlib.pyplot as plt
import other.config as config
from other.utils import netcdfToNumpy

class CMIPDataset(T.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx][:-1]
        y = self.data[idx][-1]
        return X,y


class Net(T.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hid1 = T.nn.Linear(5,10) 
    self.hid2 = T.nn.Linear(10, 10)
    # self.drop1 = T.nn.Dropout(0.50) #example of dropout layer
    self.oupt = T.nn.Linear(10, 1)

  def forward(self, x):
    z = T.relu(self.hid1(x))
    z = T.relu(self.hid2(z))
    # z = self.drop1(z)
    z = self.oupt(z)  # no activation bc of regression
    return z






# To summarize, when calling a PyTorch neural network to compute output during training, you should set the mode as net.train() 
# and not use the no_grad() statement. But when you aren't training, you should set the mode as net.eval() and use the no_grad() statement


if __name__ == "__main__":
    data = np.genfromtxt(f'{config.CESM_PATH}/cesm_data.csv',delimiter=',')
    ds = CMIPDataset(data)
    ds.data = (ds.data - ds.data.mean()) / ds.data.std() #where should this be done? 

    ds = T.utils.data.Subset(ds, list(range(0,100)))
    train_set_size = int(len(ds) * 0.8)
    valid_set_size = len(ds) - train_set_size
    train,test = T.utils.data.random_split(ds, [train_set_size, valid_set_size], generator=T.Generator().manual_seed(42))

    model = Net()
    loss_function = nn.MSELoss()
    learning_rate = 0.05
    train_ldr = T.utils.data.DataLoader(train,batch_size=1,shuffle=True)
    test_ldr = T.utils.data.DataLoader(test,batch_size=1,shuffle=True)
    optimizer = T.optim.SGD(model.parameters(), lr=learning_rate)
    losses = []
    # X_train = T.from_numpy(X_train).float()
    # y_train = T.from_numpy(y_train).float()
    # X_test = T.from_numpy(X_test).float()
    # y_test = T.utils.data.Dataloader(T.from_numpy(y_test).float()
    min_valid_loss = np.inf
    for epoch in range(2):
        train_loss = 0
        model.train()
        for X,y in train_ldr:
            pred_y = model(X.float())[0]
            print(X,pred_y,y)
            loss = loss_function(pred_y, y.float().unsqueeze(1))
            optimizer.zero_grad() #clears old gradients from previous steps 
            loss.backward() #compute gradient
            optimizer.step() #take step based on gradient
            train_loss += loss.item()
            print(loss.item())
        losses.append(train_loss)


        model.eval()
        valid_loss = 0
        for X,y in test_ldr:
            pred_y = model(X.float())
            test_loss = loss_function(pred_y,y.float())
            valid_loss += test_loss.item() * X.size(0)

        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})')
            min_valid_loss = valid_loss
    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Learning rate %f"%(learning_rate))
    plt.show()

