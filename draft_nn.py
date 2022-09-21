import numpy as np
import torch as T
import torch.nn as nn
import matplotlib.pyplot as plt
from other import config
from sklearn import preprocessing
import time
class CMIPDataset(T.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx][-8:-4]
        y = self.data[idx][-4:-3]
        return X,y


class Net(T.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hid1 = T.nn.Linear(4,5) 
    self.hid2 = T.nn.Linear(5,10)
    self.hid3 = T.nn.Linear(10,5)
    # self.drop1 = T.nn.Dropout(0.50) #example of dropout layer
    self.oupt = T.nn.Linear(5, 1)

  def forward(self, x):
    z = T.relu(self.hid1(x))
    z = T.relu(self.hid2(z))
    z = T.relu(self.hid3(z)) 
    # z = self.drop1(z)
    z = self.oupt(z)  # no activation bc of regression
    return z


# To summarize, when calling a PyTorch neural network to compute output during training, you should set the mode as net.train() 
# and not use the no_grad() statement. But when you aren't training, you should set the mode as net.eval() and use the no_grad() statement


if __name__ == "__main__":
    data = np.genfromtxt(f'{config.CESM_PATH}/cesm_data.csv',delimiter=',')
    
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    data = min_max_scaler.fit_transform(data)
    ds = data[data[:,-3] < 1]
    final_test = data[data[:,-3] == 1] 
    
    # for col_index in range(len(ds[0,:])):
    #     norm_col = min_max_scaler.fit_transform(ds[:,col_index].reshape(-1,1))
    #     ds[:,col_index] = norm_col.reshape(1,-1)
    # ds = min_max_scaler.fit_transform(ds)
    ds = CMIPDataset(ds)
    # for col_index in range(len(final_test[0,:])):
    #     norm_col = min_max_scaler.fit_transform(final_test[:,col_index].reshape(-1,1))
    #     final_test[:,col_index] = norm_col.reshape(1,-1)
    # final_test = min_max_scaler.fit_transform(final_test)
    final_test = CMIPDataset(final_test)
    # ds = T.utils.data.Subset(ds, list(range(0,10)))
    train_set_size = int(len(ds) * 0.8)
    valid_set_size = len(ds) - train_set_size
    train,test = T.utils.data.random_split(ds, [train_set_size, valid_set_size], generator=T.Generator().manual_seed(0))

    model = Net()
    loss_function = nn.MSELoss()
    learning_rate = 0.00001 #any higher than this does not optimize
    train_ldr = T.utils.data.DataLoader(train,batch_size=100,shuffle=True)
    test_ldr = T.utils.data.DataLoader(test,batch_size=100,shuffle=True)
    final_ldr = T.utils.data.DataLoader(final_test,batch_size=1,shuffle=False)
    optimizer = T.optim.SGD(model.parameters(), lr=learning_rate)
    losses = []
    min_valid_loss = np.inf
    for epoch in range(100):
        start_time = time.time()
        train_loss = 0
        model.train()
        for X,y in train_ldr:
            # print(X.float())
            # print(y[0])
            # print(X.data.numpy(),y.data.numpy())
            pred_y = model(X.float())
            # print('pred_y',pred_y)
            loss = loss_function(pred_y, y.float())
            optimizer.zero_grad() #clears old gradients from previous steps 
            loss.backward() #compute gradient
            optimizer.step() #take step based on gradient
            train_loss += loss.item()
            # print('loss item',loss.item())
            # print(pred_y.data.numpy())
        losses.append(train_loss)


        model.eval()
        valid_loss = 0
        for X,y in test_ldr:
            # print(X.data.numpy(),y.data.numpy())
            pred_y = model(X.float())
            # print(pred_y.data.numpy())
            test_loss = loss_function(pred_y,y.float())
            valid_loss += test_loss.item() * X.size(0)

        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {valid_loss}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f})')
            min_valid_loss = valid_loss
        print(f'Epoch finished in {time.time() - start_time} seconds')

    final_loss = 0
    for X,y in final_ldr:
        print(X.data.numpy(),y.data.numpy())
        final_pred = model(X.float())
        loss = loss_function(final_pred,y.float())
        final_loss += loss.item() * X.size(0)
        print('prediction:',final_pred.data.numpy())
    # print(final_loss)
    # plt.plot(losses)
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.title("Learning rate %f"%(learning_rate))
    # plt.show()



