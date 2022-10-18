import numpy as np
import torch as T
import torch.nn as nn
import matplotlib.pyplot as plt
from other import config
from sklearn import preprocessing
import time
from pickle import dump
from sklearn.metrics import mean_squared_error, r2_score
from torchmetrics.functional import r2_score as R2Score
import pandas as pd


class CMIPDataset(T.utils.data.Dataset):
    def __init__(self, data,num_of_inputs):
        self.data = data
        self.num_of_inputs = num_of_inputs
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx][:self.num_of_inputs]
        y = self.data[idx][-4:]
        return X,y


class Net(T.nn.Module):
  def __init__(self,num_of_inputs):
    super(Net, self).__init__()
    self.hid1 = T.nn.Linear(num_of_inputs,20) 
    self.hid2 = T.nn.Linear(20,30)
    self.hid5 = T.nn.Linear(30,20)
    # self.drop1 = T.nn.Dropout(0.50) #example of dropout layer
    self.oupt = T.nn.Linear(20, 4)

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
   
    
    
    #prepare and scale data
    data = pd.read_csv(f'{config.DATA_PATH}/cesm_data.csv')
    data['grass_crop_shrub'] = data['cropFrac'] + data['grassFrac'] + data['shrubFrac']
    ds = data[data['years'] < 2012]
    final_test = data[data['years'] >= 2012] 

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    input_variables = ['pr','tas','# lai','treeFrac','baresoilFrac','ps','grass_crop_shrub']
    input_variable_tuple = ('pr','tas','# lai','treeFrac','baresoilFrac','ps','grass_crop_shrub')
    target_variables = ['cSoil','cCwd','cVeg','cLitter']
    scaler = min_max_scaler.fit(ds.loc[:,input_variables])
    ds.loc[:,input_variable_tuple] = scaler.transform(ds.loc[:,input_variable_tuple])
    final_test.loc[:,input_variable_tuple] = scaler.transform(final_test.loc[:,input_variable_tuple])
    ds = CMIPDataset(ds[input_variables + target_variables].to_numpy(),num_of_inputs=len(input_variables))
    final_test = CMIPDataset(final_test[input_variables + target_variables].to_numpy(),num_of_inputs=len(input_variables))

    #hyperparameters
    num_of_epochs = 100
    learning_rate = 0.0001
    test_validation_batch_size = 100
    #split train and validation
    train_set_size = int(len(ds) * 0.8)
    valid_set_size = len(ds) - train_set_size
    train,test = T.utils.data.random_split(ds, [train_set_size, valid_set_size], generator=T.Generator().manual_seed(0))

    model = Net(len(input_variables))
    loss_function = nn.MSELoss()
    train_ldr = T.utils.data.DataLoader(train,batch_size=test_validation_batch_size,shuffle=True)
    test_ldr = T.utils.data.DataLoader(test,batch_size=test_validation_batch_size,shuffle=True)
    final_ldr = T.utils.data.DataLoader(final_test,batch_size=1,shuffle=False)
    optimizer = T.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    min_valid_loss = np.inf
    for epoch in range(num_of_epochs):
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
    y_pred_list = []
    y_target = []
    model.eval()
    for X,y in final_ldr:
        final_pred = model(X.float())
        loss = loss_function(final_pred,y.float())
        final_loss += loss.item() * X.size(0)
        total = total + y.float()
        pred_total += final_pred
        y_pred_list.append(final_pred.detach().numpy()[0])
        y_target.append(y.detach().numpy()[0])
    print(f'mse: {mean_squared_error(y_pred_list,y_target)}')
    print(f'total r2 score {r2_score(y_target,y_pred_list)}')
    print(f'individual r2 {R2Score(T.tensor(np.array(y_pred_list)),T.tensor(np.array(y_target)),multioutput="raw_values")}')
    print(final_loss,total,pred_total)
    T.save(model.state_dict(), '/Users/gclyne/thesis/data/trained_net')
    dump(scaler, open('/Users/gclyne/thesis/data/scaler.pkl', 'wb'))
    # print(final_loss)
    # plt.plot(losses)
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.title("Learning rate %f"%(learning_rate))
    # plt.show()



