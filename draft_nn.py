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
        X = self.data[idx][:4]
        y = self.data[idx][-8:-4]
        return X,y


class Net(T.nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.hid1 = T.nn.Linear(4,10) 
    self.hid2 = T.nn.Linear(10,20)
    self.hid5 = T.nn.Linear(20,10)
    # self.drop1 = T.nn.Dropout(0.50) #example of dropout layer
    self.oupt = T.nn.Linear(10, 4)

  def forward(self, x):
    z = T.relu(self.hid1(x))
    z = T.relu(self.hid2(z))
    z = T.relu(self.hid5(z))

    # z = self.drop1(z)
    z = self.oupt(z)  # no activation bc of regression
    return z


# To summarize, when calling a PyTorch neural network to compute output during training, you should set the mode as net.train() 
# and not use the no_grad() statement. But when you aren't training, you should set the mode as net.eval() and use the no_grad() statement


if __name__ == "__main__":
    data = np.genfromtxt(f'{config.CESM_PATH}/cesm_data.csv',delimiter=',',skip_header=1)
    
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))

    ds = data[data[:,-3] < 2012]
    final_test = data[data[:,-3] == 2012] 

    scaler = min_max_scaler.fit(ds[:,[-4]])
    ds[:,[-5]] = scaler.transform(ds[:,[-5]])
    ds[:,[-6]] = scaler.transform(ds[:,[-6]])
    ds[:,[-7]] = scaler.transform(ds[:,[-7]])
    ds[:,[-8]] = scaler.transform(ds[:,[-8]])
    ds[:,[-4]] = scaler.transform(ds[:,[-4]])
    ds[:,[0]] = min_max_scaler.fit_transform(ds[:,[0]])
    ds[:,[1]] = min_max_scaler.fit_transform(ds[:,[1]])
    ds[:,[2]] = min_max_scaler.fit_transform(ds[:,[2]])
    ds[:,[3]] = min_max_scaler.fit_transform(ds[:,[3]])

    ds = np.concatenate((ds[:,:4],ds[:,-8:-4]),1)
    print(ds)
    scaler = min_max_scaler.fit(final_test[:,[-4]])
    final_test[:,[-5]] = scaler.transform(final_test[:,[-5]])
    final_test[:,[-6]] = scaler.transform(final_test[:,[-6]])
    final_test[:,[-7]] = scaler.transform(final_test[:,[-7]])
    final_test[:,[-8]] = scaler.transform(final_test[:,[-8]])
    final_test[:,[-4]] = scaler.transform(final_test[:,[-4]])
    final_test[:,[0]] = min_max_scaler.fit_transform(final_test[:,[0]])
    final_test[:,[1]] = min_max_scaler.fit_transform(final_test[:,[1]])
    final_test[:,[2]] = min_max_scaler.fit_transform(final_test[:,[2]])
    final_test[:,[3]] = min_max_scaler.fit_transform(final_test[:,[3]])
    final_test = np.concatenate((final_test[:,:4],final_test[:,-8:-4]),1)

    ds = CMIPDataset(ds)
    final_test = CMIPDataset(final_test)

    # ds = T.utils.data.Subset(ds, list(range(0,10)))
    train_set_size = int(len(ds) * 0.8)
    valid_set_size = len(ds) - train_set_size
    train,test = T.utils.data.random_split(ds, [train_set_size, valid_set_size], generator=T.Generator().manual_seed(0))

    model = Net()
    loss_function = nn.MSELoss()
    learning_rate = 0.00001
    train_ldr = T.utils.data.DataLoader(train,batch_size=100,shuffle=True)
    test_ldr = T.utils.data.DataLoader(test,batch_size=100,shuffle=True)
    final_ldr = T.utils.data.DataLoader(final_test,batch_size=1,shuffle=False)
    optimizer = T.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    print(model.parameters())
    min_valid_loss = np.inf
    for epoch in range(150):
        start_time = time.time()
        train_loss = 0
        model.train()
        for X,y in train_ldr:
            pred_y = model(X.float())
            loss = loss_function(pred_y, y.float())
            optimizer.zero_grad() #clears old gradients from previous steps 
            loss.backward() #compute gradient
            optimizer.step() #take step based on gradient
            train_loss += loss.item()
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
    total = 0
    pred_total = 0
    for X,y in final_ldr:
        print(X.data.numpy(),y.data.numpy())
        final_pred = model(X.float())
        loss = loss_function(final_pred,y.float())
        final_loss += loss.item() * X.size(0)
        total = total + y.float()
        pred_total += final_pred
        print('prediction:',final_pred.data.numpy())
    print(final_loss,total,pred_total)
    # print(final_loss)
    # plt.plot(losses)
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.title("Learning rate %f"%(learning_rate))
    # plt.show()



